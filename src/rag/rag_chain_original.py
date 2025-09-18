from typing import List, Dict, Any, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import settings
from src.vectorstore import VectorDatabase
from src.models import ModelRegistry
from src.utils import TextProcessor
from src.constants import (
    MODEL_PROMPT_TOKENS, MODEL_MIN_CONTEXT, MODEL_MAX_CONTEXT,
    MODEL_SAFETY_MARGIN, LOCAL_MODEL_SAFETY_MARGIN, TOKEN_TO_CHAR_RATIO,
    OPTIMAL_DOC_LENGTH_RANGE, MAX_DOC_LENGTH_SCORE, LOCAL_MODEL_MAX_DOCUMENTS,
    REQUEST_TIMEOUT, LOCAL_MODEL_TIMEOUT, MAX_RETRIES, RETRY_DELAY, RETRY_DELAY_LOCAL
)
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# 새로운 컨텍스트 분할 모듈 임포트
from .context_chunker import ContextChunker, ContextChunk
from .summarizer import HierarchicalSummarizer
from .rag_parallel_processor import ParallelRAGProcessor

# 로거 설정
logger = logging.getLogger(__name__)

# 성능 유틸
try:
    from src.utils.perf import now
except Exception:
    # 유틸 불가 시 표준 time 대체
    import time

    def now() -> float:
        return time.perf_counter()

class RAGChain:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, vector_db: Optional[VectorDatabase] = None):
        # 벡터 DB 인스턴스를 외부에서 주입받거나 새로 생성
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDatabase()
        
        # 로컬 GGUF 세부 동작 제어 플래그 (Qwen/Midm 특화 프롬프트/샘플링)
        self.is_qwen_gguf = False
        self.is_midm_gguf = False

        self.llm = self._initialize_llm(provider, model)
        self.streaming_llm = self._initialize_streaming_llm(provider, model)  # 스트리밍용 LLM
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
        self.streaming_chain = self._create_streaming_chain()  # 스트리밍용 체인
        self.current_provider = provider or settings.llm_provider
        self.current_model = model
        
        # 새로운 컨텍스트 분할 시스템 초기화
        self.context_chunker = ContextChunker(max_chunk_size=30000)  # 30KB 청크
        self.summarizer = HierarchicalSummarizer(llm=self.llm, max_summary_length=500)
        self.parallel_processor = ParallelRAGProcessor(
            max_workers=3,
            chunk_timeout=30.0,
            enable_result_merging=True
        )
        self.parallel_processor.set_summarizer(self.summarizer)
        
        # 대량 문서 처리 활성화 여부
        self.enable_large_context_processing = getattr(settings, 'enable_large_context_processing', True)
        self.large_context_threshold = getattr(settings, 'large_context_threshold', 50000)  # 50KB 임계값
        
        # 모델 레지스트리에서 설정 가져오기
        if settings.llm_provider == "local":
            ModelRegistry.update_local_model_config(
                settings.local_llm_max_tokens,
                settings.local_llm_context_window
            )
            # 로컬 기본 모델명을 일관되게 사용 (혼란 방지)
            settings.local_llm_model = "local-model"
    
    def _initialize_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """일반 LLM 초기화 (비스트리밍)"""
        return self._create_llm(provider, model, streaming=False)
    
    def _initialize_streaming_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """스트리밍 LLM 초기화"""
        return self._create_llm(provider, model, streaming=True)
    
    def _create_llm(self, provider: Optional[str] = None, model: Optional[str] = None, streaming: bool = False):
        """LLM 생성 (스트리밍/비스트리밍 공통)"""
        # provider가 지정되지 않으면 설정에서 가져옴
        if provider is None:
            provider = settings.llm_provider
        
        # 실제 사용할 모델명 결정
        if provider == "local":
            actual_model = model or settings.local_llm_model
        else:
            actual_model = model or getattr(settings, f"{provider}_model", None)
        
        # 모델별 최대 토큰 수 가져오기 (로컬은 설정값 우선)
        max_tokens = self._get_max_tokens_for_model(provider, actual_model) if actual_model else settings.max_tokens
        if provider == "local":
            try:
                max_tokens = int(getattr(settings, 'local_llm_max_tokens', max_tokens))
            except Exception:
                pass
        
        logger.info(f"LLM 초기화: provider={provider}, model={actual_model}, max_tokens={max_tokens}, streaming={streaming}")
        
        # 스트리밍 콜백 설정
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
            
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            return ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                callbacks=callbacks
            )
        
        elif provider == "google":
            if not settings.google_api_key:
                raise ValueError("Google API 키가 설정되지 않았습니다.")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=actual_model,
                temperature=settings.temperature,
                max_output_tokens=max_tokens,
                streaming=streaming,
                callbacks=callbacks
            )
        
        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
            return ChatAnthropic(
                anthropic_api_key=settings.anthropic_api_key,
                model_name=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                callbacks=callbacks
            )
        
        elif provider == "local":
            """로컬 모델: OpenAI 호환 HTTP 엔드포인트 사용.
            - 엔드포인트: {base}/v1
            - 헬스체크: GET {base}/health
            - 큐 상태: GET {base}/v1/queue/stats
            - 모델 목록: GET {base}/v1/models
            - 채팅: POST {base}/v1/chat/completions
            - 임베딩: POST {base}/v1/embeddings
            """
            from src.constants import LOCAL_MODEL_TIMEOUT
            # 모델에 엔드포인트가 포함된 경우(형식: "<model_id>|<base_url>") 우선 사용
            override_base = None
            if actual_model and isinstance(actual_model, str) and "|" in actual_model:
                try:
                    parts = actual_model.split("|", 1)
                    actual_model = parts[0]
                    override_base = parts[1]
                except Exception:
                    override_base = None
            # 스킴 보강 및 프리플라이트 점검
            def ensure_scheme(u: str) -> str:
                if not u:
                    return u
                u = u.strip()
                if not (u.startswith("http://") or u.startswith("https://")):
                    return "http://" + u
                return u
            base = ensure_scheme(override_base or settings.local_llm_base_url).rstrip("/")
            try:
                from src.utils.http_probe import probe_local_server
                probe = probe_local_server(base)
                logger.info(f"로컬 LLM 프리플라이트: base={base} health={probe['health']} models={probe['models']} queue={probe['queue']}")
                # 모델 엔드포인트가 응답하지 않으면 1620으로 폴백 시도(멀티모달/기본 서버)
                if not probe.get('models'):
                    host = base.split('://', 1)[-1].split(':')[0]
                    cand = f"http://{host}:1620"
                    probe2 = probe_local_server(cand)
                    if probe2.get('models'):
                        logger.warning(f"/v1/models 미응답: {base} → {cand} 폴백")
                        base = cand
            except Exception:
                # 점검 실패 시 그대로 진행
                pass
            logger.info(f"로컬 LLM HTTP 엔드포인트 사용: {base} (model={actual_model})")
            logger.info(f"ChatOpenAI(local) 설정: base_url={base}/v1, model={actual_model}")
            return ChatOpenAI(
                api_key=settings.local_llm_api_key,
                base_url=f"{base}/v1",
                model=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                callbacks=callbacks,
                timeout=LOCAL_MODEL_TIMEOUT,
                max_retries=getattr(settings, 'local_llm_max_retries', 3),
            )
        
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성 (Qwen GGUF는 ChatML 형식 사용)"""
        if getattr(self, 'is_qwen_gguf', False):
            template = (
                "<|im_start|>system\n"
                "역할: 한국 역사 주제를 한국어로 쉽고 정확하게 설명하는 조력자.\n"
                "원칙: (1) 사실 근거 (2) 간결하고 쉬운 표현 (3) 번호/불릿으로 정리 (4) 한자/전문 용어는 괄호로 풀어쓰기.\n"
                "중요: 아래 원칙이나 지침 문구를 답변에 출력하지 말 것. '초등학생 수준' 등 메타 문구 금지.\n"
                "출력 형식: 질문에 대한 답변 본문만. 도입 멘트(예: '~설명해줄게요')와 예시/지침 제목 출력 금지.\n\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "다음은 검색된 관련 문서들입니다:\n{context}\n\n"
                "위 문서들을 참고하여 다음 질문에 답변하세요.\n질문: {question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            template = """당신은 국사의 전문적인 지식을 초등학교 학생들도 알기쉽게 친절하게 전달하는 AI 어시스턴트입니다.
상세하고 정확한 정보를 제공하면서도, 초등학생들도 이해할 수 있도록 쉽게 설명하는 것이 당신의 역할입니다.

다음은 검색된 관련 문서들입니다:
{context}

위 문서들의 내용을 참고하여 다음 질문에 답변해주세요:
{question}

답변:"""

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _create_chain(self):
        """일반 LLM 체인 생성"""
        # 새로운 LCEL 방식 사용
        return self.prompt_template | self.llm
    
    def _create_streaming_chain(self):
        """스트리밍 LLM 체인 생성"""
        return self.prompt_template | self.streaming_llm
    
    def _get_model_max_tokens(self, model_id: str) -> int:
        """모델별 최대 토큰 수 반환"""
        return ModelRegistry.get_max_tokens(model_id, default=settings.max_tokens)
    
    def _get_model_context_window(self, model_id: str) -> int:
        """모델별 전체 컨텍스트 윈도우 크기 반환 (토큰 단위)"""
        return ModelRegistry.get_context_window(model_id, default=8192)
    
    def _get_max_tokens_for_model(self, provider: str, model: str) -> int:
        """특정 모델의 최대 토큰 수 반환"""
        return self._get_model_max_tokens(model)
    
    def _get_max_context_length_for_model(self, provider: str = None, model: str = None) -> int:
        """현재 모델의 최대 컨텍스트 길이 계산 (문자 단위)"""
        # 현재 모델 정보 사용
        if not provider:
            provider = self.current_provider
        if not model:
            model = self.current_model or getattr(settings, f"{provider}_model", None)
        
        # 모델의 전체 컨텍스트 윈도우 크기 (토큰)
        total_context_tokens = self._get_model_context_window(model)
        
        # 출력용 토큰 예약
        output_tokens = self._get_max_tokens_for_model(provider, model)
        
        # 프롬프트 템플릿용 토큰 예약
        prompt_tokens = MODEL_PROMPT_TOKENS.get(provider, MODEL_PROMPT_TOKENS["default"])
        
        # 안전 마진
        safety_margin = LOCAL_MODEL_SAFETY_MARGIN if provider == "local" else MODEL_SAFETY_MARGIN
        
        # 사용 가능한 컨텍스트 토큰
        available_context_tokens = int((total_context_tokens - output_tokens - prompt_tokens) * safety_margin)
        
        # 토큰을 문자로 변환
        max_context_chars = available_context_tokens * TOKEN_TO_CHAR_RATIO
        
        # 최소/최대 제한
        min_context = MODEL_MIN_CONTEXT.get(provider, MODEL_MIN_CONTEXT["default"])
        max_context = MODEL_MAX_CONTEXT["default"]
        
        # 모델별 특별 제한
        # 로컬 모델은 설정/레지스트리 기반 계산값을 그대로 활용 (별도 상수 제한 제거)
        if provider == "openai" and model and "gpt-3.5" in model:
            max_context = MODEL_MAX_CONTEXT["gpt-3.5"]
        
        return max(min_context, min(max_context, max_context_chars))
    
    def _format_documents(self, documents: List[tuple], query: str = "") -> str:
        """
        검색된 문서를 컨텍스트로 포맷
        - 중복 제거 및 재랭킹
        - 컨텍스트 최적화
        - 관련성 점수 기반 정렬
        """
        if not documents:
            return ""
        
        # 1. 중복 문서 제거 (유사한 내용 필터링)
        unique_documents = self._remove_duplicate_documents(documents)
        
        # 2. 재랭킹 (관련성과 다양성 모두 고려)
        reranked_documents = self._rerank_documents(unique_documents)
        
        # 3. 컨텍스트 최적화
        context_parts = []
        total_length = 0
        max_context_length = self._get_max_context_length_for_model()  # 모델별 동적 컨텍스트 길이
        
        logger.info(f"현재 모델의 최대 컨텍스트 길이: {max_context_length:,}자")
        
        for i, (doc, score) in enumerate(reranked_documents):
            # 관련성 점수 표시 형태 개선
            if settings.vector_db_type == "faiss":
                # FAISS는 거리 기반 (낮을수록 좋음)
                relevance_percent = max(0, min(100, (2.0 - float(score)) * 50))
            else:
                # ChromaDB는 유사도 기반 (높을수록 좋음)
                relevance_percent = min(100, float(score) * 100)
            
            source = doc.metadata.get('source', '알 수 없음')
            file_name = doc.metadata.get('file_name', '알 수 없음')
            chunk_info = doc.metadata.get('chunk_id', f'chunk_{i}')
            content = self._optimize_content(doc.page_content.strip(), query)
            
            # 컨텍스트 길이 체크
            content_preview = f"[문서 {i+1}: {file_name} (관련도: {relevance_percent:.1f}%, {chunk_info})]\n{content}\n"
            
            # 컨텍스트 길이 제한을 더 유연하게 처리
            if total_length + len(content_preview) > max_context_length:
                # 최소한 5개 문서는 포함하도록 보장
                if len(context_parts) < 5:
                    context_parts.append(content_preview)
                    total_length += len(content_preview)
                else:
                    logger.info(f"컨텍스트 길이 제한으로 {len(reranked_documents) - i}개 문서 생략")
                    break
            else:
                context_parts.append(content_preview)
                total_length += len(content_preview)
        
        # 4. 최종 컨텍스트 구성
        final_context = "\n" + "="*50 + "\n".join(context_parts) + "="*50 + "\n"
        
        logger.info(f"컨텍스트 구성 완료: {len(context_parts)}개 문서, {total_length}자")
        return final_context
    
    def _remove_duplicate_documents(self, documents: List[tuple]) -> List[tuple]:
        """중복 및 유사한 문서 제거"""
        if len(documents) <= 1:
            return documents
        
        unique_docs = []
        seen_contents = set()
        
        for doc, score in documents:
            content = doc.page_content.strip()
            
            # 완전 중복 체크
            content_hash = hash(content)
            if content_hash in seen_contents:
                continue
            
            # 유사도가 매우 높은 문서 체크 (간단한 중복 감지)
            is_similar = False
            for existing_content in seen_contents:
                if self._calculate_content_similarity(content, str(existing_content)) > 0.9:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_docs.append((doc, score))
                seen_contents.add(content_hash)
        
        logger.info(f"중복 제거: {len(documents)} -> {len(unique_docs)}개 문서")
        return unique_docs
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """두 컨텐츠 간의 유사도 계산 (간단한 Jaccard 유사도)"""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _rerank_documents(self, documents: List[tuple]) -> List[tuple]:
        """
        문서 재랭킹
        - 관련성 점수 기반
        - 문서 품질 고려
        - 다양성 확보
        """
        if len(documents) <= 1:
            return documents
        
        scored_documents = []
        
        for doc, original_score in documents:
            # 1. 원본 검색 점수
            relevance_score = self._normalize_score(original_score)
            
            # 2. 문서 품질 점수
            quality_score = self._calculate_document_quality(doc)
            
            # 3. 컨텐츠 길이 점수 (너무 짧거나 긴 문서 패널티)
            length_score = self._calculate_length_score(doc.page_content)
            
            # 4. 메타데이터 품질 점수
            metadata_score = self._calculate_metadata_quality(doc.metadata)
            
            # 5. 종합 점수 계산 (관련성에 더 높은 가중치)
            final_score = (
                relevance_score * 0.7 +
                quality_score * 0.1 +
                length_score * 0.1 +
                metadata_score * 0.1
            )
            
            scored_documents.append((doc, original_score, final_score))
        
        # 종합 점수로 정렬 (높을수록 좋음)
        scored_documents.sort(key=lambda x: x[2], reverse=True)
        
        # 원본 형태로 변환
        reranked = [(doc, score) for doc, score, _ in scored_documents]
        
        logger.info("문서 재랭킹 완료")
        return reranked
    
    def _normalize_score(self, score: float) -> float:
        """점수 정규화 (0-1 범위로)"""
        if settings.vector_db_type == "faiss":
            # FAISS 거리를 유사도로 변환
            return max(0, min(1, (2.0 - float(score)) / 2.0))
        else:
            # ChromaDB 유사도 그대로 사용
            return min(1, max(0, float(score)))
    
    def _calculate_document_quality(self, doc: Document) -> float:
        """문서 품질 점수 계산"""
        content = doc.page_content
        quality_score = 0.5  # 기본 점수
        
        # 한국어 문장 완성도 체크
        korean_sentences = len([s for s in content.split('.') if any('\u3131' <= c <= '\u3163' or '\uac00' <= c <= '\ud7a3' for c in s)])
        if korean_sentences > 0:
            quality_score += 0.2
        
        # 구조화된 내용 체크 (번호 매기기, 목록 등)
        if any(pattern in content for pattern in ['1.', '2.', '•', '-', '가.', '나.']):
            quality_score += 0.1
        
        # 전문 용어 및 키워드 포함 여부
        professional_terms = ['시스템', '정보', '구축', '운영', '관리', '지침', '규정', '법령',
                            '발굴', '유물', '토기', '백제', '고구려', '출토', '유적', '조사',
                            '파수', '고배', '호', '옹', '시루', '토성', '몽촌토성']
        term_count = sum(1 for term in professional_terms if term in content)
        quality_score += min(0.2, term_count * 0.05)
        
        return min(1.0, quality_score)
    
    def _calculate_length_score(self, content: str) -> float:
        """컨텐츠 길이 점수 계산"""
        length = len(content)
        
        # 최적 길이 범위
        min_optimal, max_optimal = OPTIMAL_DOC_LENGTH_RANGE
        
        if min_optimal <= length <= max_optimal:
            return 1.0
        elif min_optimal // 2 <= length < min_optimal or max_optimal < length <= max_optimal * 1.5:
            return 0.8
        elif min_optimal // 4 <= length < min_optimal // 2 or max_optimal * 1.5 < length <= MAX_DOC_LENGTH_SCORE:
            return 0.6
        else:
            return 0.3
    
    def _calculate_metadata_quality(self, metadata: dict) -> float:
        """메타데이터 품질 점수 계산"""
        quality_score = 0.5
        
        # 파일명이 의미있는지 체크
        file_name = metadata.get('file_name', '')
        if file_name and not file_name.startswith('temp_') and len(file_name) > 5:
            quality_score += 0.2
        
        # 청크 정보가 있는지 체크
        if 'chunk_id' in metadata:
            quality_score += 0.1
        
        # 처리 방법 정보가 있는지 체크
        if 'processing_method' in metadata:
            quality_score += 0.1
        
        # 추출 방법 정보 (OCR vs 일반)
        extraction_method = metadata.get('extraction_method', '')
        if extraction_method == 'pypdf':
            quality_score += 0.1  # 일반 추출이 더 안정적
        
        return min(1.0, quality_score)
    
    def _optimize_content(self, content: str, query: str = "") -> str:
        """컨텐츠 최적화 및 오염 제거.
        - 메타 지침/예시/ChatML 토큰 제거(문서 내 포함된 가이드 문구가 답변에 스며드는 현상 방지)
        """
        # 1) 공백 정리(과하지 않게)
        text = re.sub(r"\s+", " ", content).strip()

        # 2) ChatML/역할 토큰 제거
        blacklist_tokens = ["<|im_start|>", "<|im_end|>", "\nuser ", "\nassistant ", "\nsystem "]
        for t in blacklist_tokens:
            text = text.replace(t, " ")

        # 3) 라인 단위로 지침/예시 문구 필터
        banned_patterns = [
            r"^\s*답변\s*시\s*지켜야\s*할\s*규칙.*$",
            r"^\s*이런\s*식으로\s*답변.*$",
            r"^\s*답변\s*예시.*$",
            r"^\s*예시.*$",
        ]
        lines = [ln for ln in re.split(r"\s*\n\s*", content) if ln.strip()]
        filtered = []
        for ln in lines:
            if any(re.search(p, ln, flags=re.IGNORECASE) for p in banned_patterns):
                continue
            if ln.strip() in ("user", "assistant", "system"):
                continue
            filtered.append(ln)
        text = "\n".join(filtered) if filtered else text

        return text

    def _sanitize_output_chunk(self, text: str) -> str:
        """스트리밍 출력 중 메타/토큰 제거(가벼운 필터)."""
        if not text:
            return text
        # 토큰 제거
        text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        # 한 줄 지침/예시 라인 제거
        banned_fragments = [
            "답변 시 지켜야 할 규칙",
            "이런 식으로 답변",
            "답변 예시",
        ]
        if any(fr in text for fr in banned_fragments):
            lines = text.splitlines()
            kept = [ln for ln in lines if not any(fr in ln for fr in banned_fragments)]
            text = "\n".join(kept)
        # 역할 태그 라인 제거
        role_prefixes = ("user ", "assistant ", "system ")
        lines = text.splitlines()
        kept2 = [ln for ln in lines if not ln.strip().lower().startswith(role_prefixes)]
        return "\n".join(kept2)
    
    def get_last_context_tokens(self) -> int:
        """마지막 쿼리에서 사용된 컨텍스트 토큰 수 반환"""
        return getattr(self, '_last_context_tokens', 0)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        향상된 질문 처리 (대량 문서 지원)
        - 쿼리 전처리 및 확장
        - 검색 결과 분석
        - 컨텍스트 분할 및 병렬 처리 (필요시)
        - 컨텍스트 최적화
        """
        try:
            t0 = now()
            # 0. 쿼리 전처리 및 확장
            processed_question = self._preprocess_query(question)
            
            # 벡터 DB 상태 확인
            doc_count = self.vector_db.get_document_count()
            logger.info(f"벡터 DB 문서 수: {doc_count}")
            
            if doc_count == 0:
                return {
                    "answer": "벡터 데이터베이스에 문서가 없습니다. 먼저 문서를 업로드해주세요.",
                    "sources": [],
                    "status": "no_documents",
                    "search_info": self._get_search_info(question, [])
                }
            
            # 1. 관련 문서 검색 (향상된 검색 사용)
            # 로컬 모델의 경우 검색 문서 수를 줄임
            k_docs = settings.k_documents
            if self.current_provider == "local":
                k_docs = min(LOCAL_MODEL_MAX_DOCUMENTS, settings.k_documents)  # 로컬 모델 제한
                logger.info(f"로컬 모델 사용 중 - 검색 문서 수를 {k_docs}개로 제한")
            
            t_search_start = now()
            relevant_docs = self.vector_db.search(processed_question, k=k_docs)
            t_search = now() - t_search_start
            logger.info(f"검색 결과: {len(relevant_docs)}개 문서")
            
            if not relevant_docs:
                # 검색 실패 시 더 관대한 검색 시도
                fallback_results = self._fallback_search(processed_question)
                
                if not fallback_results:
                    return {
                        "answer": self._generate_no_results_message(question, doc_count),
                        "sources": [],
                        "status": "no_relevant_documents",
                        "search_info": self._get_search_info(question, [])
                    }
                else:
                    relevant_docs = fallback_results
            
            # 2. 대량 문서 처리 여부 결정
            total_context_length = sum(len(doc.page_content) for doc, _ in relevant_docs)
            
            if (self.enable_large_context_processing and 
                total_context_length > self.large_context_threshold):
                # 대량 문서 처리 모드
                logger.info(f"대량 문서 처리 모드 활성화: {total_context_length:,}자")
                t_ans_start = now()
                result = self._process_large_context(question, relevant_docs)
                t_total = now() - t0
                # 경량 성능 로그
                logger.info(
                    "PERF query: provider=%s model=%s docs=%d search=%.3fs total=%.3fs",
                    self.current_provider,
                    self.current_model or getattr(settings, f"{self.current_provider}_model", None),
                    len(relevant_docs),
                    t_search,
                    t_total,
                )
                return result
            else:
                # 기존 방식 처리
                logger.info(f"기존 방식 처리: {total_context_length:,}자")
                t_ans_start = now()
                result = self._process_standard_context(question, relevant_docs)
                t_total = now() - t0
                logger.info(
                    "PERF query: provider=%s model=%s docs=%d search=%.3fs total=%.3fs",
                    self.current_provider,
                    self.current_model or getattr(settings, f"{self.current_provider}_model", None),
                    len(relevant_docs),
                    t_search,
                    t_total,
                )
                return result
            
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류 발생: {str(e)}")
            return {
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류 정보: {str(e)}",
                "sources": [],
                "status": "error",
                "search_info": {}
            }
    
    def _preprocess_query(self, query: str) -> str:
        """향상된 쿼리 전처리 및 확장"""
        if not settings.enable_query_preprocessing:
            return query
        
        # 기본 정제
        processed_query = query.strip()
        
        # 특정 고유명사가 포함된 경우 제한적 확장
        specific_terms = ['우각형파수편', '고배', '토기편', '파수편']
        contains_specific = any(term in processed_query for term in specific_terms)
        
        if settings.enable_query_expansion:
            if contains_specific:
                # 특정 용어가 있으면 기본 확장만 수행
                logger.info(f"특정 용어 감지, 제한적 확장 수행: {processed_query}")
                # 질문 의도 키워드만 추가
                if '뭐야' in processed_query or '무엇' in processed_query:
                    processed_query += " 백제 토기 유물"
                return processed_query
            else:
                # 1차: 새로운 동적 키워드 확장 사용 (임베딩 모델 포함)
                try:
                    # 임베딩 모델을 KeywordExpander에 전달
                    keyword_expander = TextProcessor.get_keyword_expander(
                        embedding_model=self.vector_db.embedding_model
                    )
                    
                    expanded_query = TextProcessor.expand_query(
                        processed_query,
                        use_static_expansion=False,  # 정적 확장은 비활성화
                        use_dynamic_expansion=True,  # 동적 확장 활성화
                        max_terms=20  # 최대 20개 키워드
                    )
                    
                    if expanded_query and expanded_query != processed_query:
                        logger.info(f"동적 키워드 확장 완료: '{query}' -> '{expanded_query[:100]}...'")
                        return expanded_query
                        
                except Exception as e:
                    logger.warning(f"동적 키워드 확장 실패: {str(e)}")
            
            # 2차: 문서 기반 관련 용어 추가 (기존 로직)
            try:
                related_terms = self.vector_db.extract_related_terms(query)
                if related_terms:
                    logger.debug(f"문서 기반 관련 용어 추출: {related_terms[:10]}")
                    
                    # 기존 쿼리에 관련 용어 추가
                    all_terms = query.split() + related_terms[:10]  # 상위 10개만 사용
                    
                    # 중복 제거 (순서 유지)
                    seen = set()
                    unique_terms = []
                    for term in all_terms:
                        if term not in seen and len(term) > 1:
                            seen.add(term)
                            unique_terms.append(term)
                    
                    processed_query = ' '.join(unique_terms[:25])  # 최대 25개 단어
                    logger.info(f"문서 기반 확장 완료: '{query}' -> '{processed_query[:100]}...'")
                    return processed_query
                    
            except Exception as e:
                logger.warning(f"문서 기반 확장 실패: {str(e)}")
            
            # 3차: 기존 정적 확장 사용 (폴백)
            try:
                processed_query = TextProcessor.expand_query(
                    processed_query, 
                    use_static_expansion=True,
                    use_dynamic_expansion=False
                )
                logger.info(f"정적 확장 사용: '{query}' -> '{processed_query[:100]}...'")
                
            except Exception as e:
                logger.warning(f"정적 확장 실패: {str(e)}")
        
        logger.info(f"쿼리 전처리 완료: '{query}' -> '{processed_query[:100]}...'")
        return processed_query
    
    def _fallback_search(self, query: str) -> List[tuple]:
        """검색 실패 시 대안 검색"""
        try:
            # 더 간단한 키워드로 검색
            simple_keywords = self._extract_keywords(query)
            if simple_keywords:
                fallback_query = " ".join(simple_keywords)
                logger.info(f"대안 검색 시도: '{fallback_query}'")
                return self.vector_db.search(fallback_query, k=5)
        except Exception as e:
            logger.error(f"대안 검색 실패: {str(e)}")
        
        return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        # TextProcessor를 사용하여 키워드 추출
        return TextProcessor.extract_keywords(query, top_k=3)
    
    def _generate_no_results_message(self, question: str, doc_count: int) -> str:
        """검색 결과 없음 메시지 생성"""
        keywords = self._extract_keywords(question)
        
        message = f"""🔍 **검색 결과가 없습니다**

현재 **{doc_count}개의 문서 청크**가 저장되어 있지만, 질문 '{question}'과 관련된 문서를 찾을 수 없었습니다.

**추출된 키워드**: {', '.join(keywords) if keywords else '없음'}

**다시 시도해보세요:**
• 다른 키워드나 표현을 사용해보세요
• 더 구체적이거나 더 일반적인 질문을 해보세요
• 문서에 실제로 포함된 내용에 대해 질문해보세요

**예시 질문:**
• "시스템이란 무엇인가요?"
• "구축 절차는 어떻게 되나요?"
• "관리 방안에 대해 알려주세요"
"""
        return message
    
    def _generate_enhanced_sources(self, documents: List[tuple]) -> List[Dict[str, Any]]:
        """향상된 출처 정보 생성"""
        sources = []
        
        for i, (doc, score) in enumerate(documents):
            # 관련성 점수 계산
            if settings.vector_db_type == "faiss":
                relevance_percent = max(0, min(100, (2.0 - float(score)) * 50))
            else:
                relevance_percent = min(100, float(score) * 100)
            
            source_info = {
                "rank": i + 1,
                "file_name": doc.metadata.get('file_name', '알 수 없음'),
                "source_path": doc.metadata.get('source', '알 수 없음'),
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i}'),
                "relevance_score": float(score),
                "relevance_percent": round(relevance_percent, 1),
                "content_length": len(doc.page_content),
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "extraction_method": doc.metadata.get('extraction_method', 'unknown'),
                "processing_method": doc.metadata.get('processing_method', 'unknown'),
                "chunk_info": f"{doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}",
                # 이미지 메타데이터 추가
                "images": doc.metadata.get('images', [])
            }
            sources.append(source_info)
        
        return sources
    
    def _get_search_info(self, question: str, documents: List[tuple]) -> Dict[str, Any]:
        """검색 정보 수집"""
        # 벡터 DB 통계 가져오기
        db_stats = self.vector_db.get_search_stats()
        
        return {
            "original_query": question,
            "processed_query": self._preprocess_query(question) if settings.enable_query_preprocessing else question,
            "total_documents_in_db": db_stats.get("total_documents", 0),
            "search_results_count": len(documents),
            "search_settings": {
                "k_documents": settings.k_documents,
                "hybrid_search": db_stats.get("hybrid_search_enabled", False),
                "mmr_search": db_stats.get("mmr_search_enabled", False),
                "vector_db_type": db_stats.get("vector_db_type", "unknown"),
                "embedding_model": db_stats.get("embedding_model", {})
            },
            "query_optimizations": {
                "preprocessing_enabled": settings.enable_query_preprocessing,
                "expansion_enabled": settings.enable_query_expansion
            }
        }
    
    def update_llm(self, provider: str, model: Optional[str] = None):
        """LLM 제공자 및 모델 변경"""
        # LLM 재초기화 (이 시점에 is_qwen_gguf / is_midm_gguf 플래그가 갱신됨)
        self.llm = self._initialize_llm(provider, model)
        self.streaming_llm = self._initialize_streaming_llm(provider, model)
        # 템플릿도 플래그에 맞춰 재생성 (Qwen/Midm 특화 템플릿 반영)
        self.prompt_template = self._create_prompt_template()
        # 체인 재생성
        self.chain = self._create_chain()
        self.streaming_chain = self._create_streaming_chain()
        self.current_provider = provider
        self.current_model = model
        
        # 현재 모델의 최대 토큰 수 및 컨텍스트 길이 로깅
        actual_model = model or getattr(settings, f"{provider}_model", None)
        max_tokens = self._get_max_tokens_for_model(provider, actual_model) if actual_model else settings.max_tokens
        max_context = self._get_max_context_length_for_model(provider, actual_model)
        logger.info(f"모델 변경 완료: {provider} - {actual_model} (max_tokens: {max_tokens}, max_context: {max_context:,}자)")
        
    def get_available_models(self) -> Dict[str, List[Dict[str, str]]]:
        """사용 가능한 모델 목록 반환"""
        models = ModelRegistry.get_all_models()

        # 로컬: 오프라인(local_models) 사용 중단하고, 서버(1620~1622 등) 모델만 노출
        try:
            server_local_models = self._get_local_models()
        except Exception:
            server_local_models = []

        models["local"] = server_local_models
        return models
    
    def _get_local_models(self) -> List[Dict[str, str]]:
        """로컬 서버(복수)에서 사용 가능한 모델 목록 조회.
        - `LOCAL_LLM_BASE_URLS`(콤마 구분) 또는 단일 `LOCAL_LLM_BASE_URL`에서 수집
        - 선택된 모델은 `model` 필드에 `"<id>|<base>"` 형식으로 base 포함
        """
        def ensure_scheme(u: str) -> str:
            u = u.strip()
            if not u:
                return u
            if not (u.startswith("http://") or u.startswith("https://")):
                return "http://" + u
            return u

        endpoints: List[str] = []
        # 1) 명시 리스트
        if getattr(settings, 'local_llm_base_urls', None):
            endpoints = [ensure_scheme(e).rstrip('/') for e in str(settings.local_llm_base_urls).split(',') if e.strip()]
        # 2) 단일 기본 + 형제 포트(1621,1622) 자동 추가
        base_default = ensure_scheme(getattr(settings, 'local_llm_base_url', '')).rstrip('/')
        if base_default and base_default not in endpoints:
            endpoints.append(base_default)
            # host 추출하여 1621/1622 추가
            try:
                from urllib.parse import urlparse
                p = urlparse(base_default)
                host = p.hostname or "210.126.109.57"
                scheme = p.scheme or "http"
                # 이미 1620인 경우 1621/1622 추가
                for port in (1621, 1622):
                    cand = f"{scheme}://{host}:{port}"
                    if cand not in endpoints:
                        endpoints.append(cand)
            except Exception:
                # 안전 폴백
                for cand in ("http://210.126.109.57:1621", "http://210.126.109.57:1622"):
                    if cand not in endpoints:
                        endpoints.append(cand)
        # 3) 최종 폴백(명시값 전혀 없을 때)
        if not endpoints:
            endpoints = [
                "http://210.126.109.57:1620",
                "http://210.126.109.57:1621",
                "http://210.126.109.57:1622",
            ]

        collected: List[Dict[str, str]] = []
        try:
            import requests  # 로컬 import로 옵셔널 의존성 처리
        except Exception:
            return collected

        for base in endpoints:
            try:
                response = requests.get(f"{base}/v1/models", timeout=10)
                if response.status_code != 200:
                    continue
                payload = response.json()
                # 다양한 스키마 대응: {data:[{id:..}]}, {models:[...]}, [..], {"object":"list","data":[..]}
                items: List[dict] = []
                if isinstance(payload, dict):
                    if isinstance(payload.get("data"), list):
                        items = payload["data"]
                    elif isinstance(payload.get("models"), list):
                        items = payload["models"]
                    else:
                        # 단일 객체 혹은 예외 스키마는 무시
                        items = []
                elif isinstance(payload, list):
                    # 문자열 목록 혹은 dict 목록
                    if all(isinstance(x, str) for x in payload):
                        items = [{"id": x} for x in payload]
                    else:
                        items = payload  # 기대: dict 리스트

                for m in items:
                    # id/name 추출 보강
                    if isinstance(m, dict):
                        mid = m.get("id") or m.get("name") or m.get("model") or "unknown"
                        owner = m.get('owned_by', m.get('owner', 'unknown'))
                    else:
                        mid = str(m)
                        owner = 'unknown'
                    port = base.split(":")[-1] if ":" in base else base
                    collected.append({
                        "name": f"{mid} ({port})",
                        "model": f"{mid}|{base}",
                        "description": f"로컬 모델 @ {base} - {owner}"
                    })
            except Exception:
                continue
        return collected
    
    def add_feedback(self, question: str, answer: str, feedback: str):
        """사용자 피드백 저장 (향후 개선을 위한 기능)"""
        # TODO: 피드백을 데이터베이스에 저장하여 모델 개선에 활용
        pass

    def stream_query(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """
        스트리밍 방식으로 질문 처리
        실시간으로 답변을 생성하여 yield 합니다.
        """
        try:
            # 0. 쿼리 전처리 및 확장
            processed_question = self._preprocess_query(question)
            
            # 벡터 DB 상태 확인
            doc_count = self.vector_db.get_document_count()
            logger.info(f"벡터 DB 문서 수: {doc_count}")
            
            if doc_count == 0:
                yield {
                    "type": "error",
                    "content": "벡터 데이터베이스에 문서가 없습니다. 먼저 문서를 업로드해주세요.",
                    "sources": [],
                    "status": "no_documents"
                }
                return
            
            # 검색 시작 알림
            yield {
                "type": "status",
                "content": "🔍 관련 문서를 검색하고 있습니다...",
                "status": "searching"
            }
            
            # 1. 관련 문서 검색
            k_docs = settings.k_documents
            if self.current_provider == "local":
                k_docs = min(LOCAL_MODEL_MAX_DOCUMENTS, settings.k_documents)
            
            relevant_docs = self.vector_db.search(processed_question, k=k_docs)
            logger.info(f"검색 결과: {len(relevant_docs)}개 문서")
            
            if not relevant_docs:
                # 대안 검색 시도
                fallback_results = self._fallback_search(processed_question)
                
                if not fallback_results:
                    yield {
                        "type": "error",
                        "content": self._generate_no_results_message(question, doc_count),
                        "sources": [],
                        "status": "no_relevant_documents"
                    }
                    return
                else:
                    relevant_docs = fallback_results
            
            # 검색 완료 및 답변 생성 시작 알림
            yield {
                "type": "status", 
                "content": f"✅ {len(relevant_docs)}개 문서 발견. 답변을 생성하고 있습니다...",
                "status": "generating"
            }
            
            # 2. 컨텍스트 생성
            context = self._format_documents(relevant_docs, question)
            self._last_context_tokens = len(context) // 4
            
            # 3. 스트리밍 방식 답변 생성
            full_response = ""
            
            # 스트리밍 체인 실행
            for chunk in self.streaming_chain.stream({
                "context": context,
                "question": question
            }):
                # 응답 텍스트 추출
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                elif isinstance(chunk, dict) and 'text' in chunk:
                    chunk_text = chunk['text']
                elif isinstance(chunk, str):
                    chunk_text = chunk
                else:
                    chunk_text = str(chunk)
                
                if chunk_text:
                    # 스트리밍 중 간단한 정화 필터 적용
                    cleaned = self._sanitize_output_chunk(chunk_text)
                    full_response += cleaned
                    yield {
                        "type": "content",
                        "content": cleaned,
                        "full_content": full_response,
                        "status": "streaming"
                    }
            
            # 4. 스트리밍 완료 후 최종 정보 전송
            sources = self._generate_enhanced_sources(relevant_docs)
            search_info = self._get_search_info(question, relevant_docs)
            
            yield {
                "type": "complete",
                "content": "",
                "full_content": full_response,
                "sources": sources,
                "status": "success",
                "search_info": search_info,
                "context_tokens": self._last_context_tokens
            }
            
        except Exception as e:
            logger.error(f"스트리밍 쿼리 처리 중 오류 발생: {str(e)}")
            yield {
                "type": "error",
                "content": f"죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류 정보: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    def _process_standard_context(self, question: str, relevant_docs: List[tuple]) -> Dict[str, Any]:
        """기존 방식의 표준 컨텍스트 처리"""
        # 2. 컨텍스트 생성 (향상된 포맷팅)
        context = self._format_documents(relevant_docs, question)
        
        # 컨텍스트 토큰 수 저장
        self._last_context_tokens = len(context) // 4
        
        # 3. LLM을 통한 답변 생성
        response = self.chain.invoke({
            "context": context,
            "question": question  # 원본 질문 사용
        })
        
        # 응답 텍스트 추출
        if hasattr(response, 'content'):
            answer_text = response.content
        elif isinstance(response, dict) and 'text' in response:
            answer_text = response['text']
        elif isinstance(response, str):
            answer_text = response
        else:
            answer_text = str(response)
        
        # 4. 출처 정보 수집 (향상된 메타데이터)
        sources = self._generate_enhanced_sources(relevant_docs)
        
        # 5. 검색 정보 수집
        search_info = self._get_search_info(question, relevant_docs)
        
        return {
            "answer": answer_text,
            "sources": sources,
            "status": "success",
            "search_info": search_info,
            "context_tokens": self._last_context_tokens,
            "processing_mode": "standard"
        }
    
    def _process_large_context(self, question: str, relevant_docs: List[tuple]) -> Dict[str, Any]:
        """대량 문서 처리를 위한 컨텍스트 분할 및 병렬 처리"""
        try:
            # 1. 컨텍스트 분할
            chunks = self.context_chunker.split_documents(
                relevant_docs, 
                question, 
                strategy="hybrid"  # 하이브리드 전략 사용
            )
            
            logger.info(f"컨텍스트 분할 완료: {len(chunks)}개 청크")
            
            # 2. 청크 순서 최적화
            optimized_chunks = self.context_chunker.optimize_chunk_order(chunks, question)
            
            # 3. 병렬 처리 실행
            if len(optimized_chunks) > 1:
                # 비동기 병렬 처리
                try:
                    # 새로운 이벤트 루프가 필요한지 확인
                    try:
                        loop = asyncio.get_running_loop()
                        # 이미 실행 중인 루프가 있는 경우 ThreadPoolExecutor 사용
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self._run_async_processing, optimized_chunks, question)
                            result = future.result(timeout=120)  # 2분 타임아웃
                    except RuntimeError:
                        # 실행 중인 루프가 없는 경우 직접 실행
                        result = asyncio.run(self._run_async_processing(optimized_chunks, question))
                        
                except Exception as e:
                    logger.warning(f"병렬 처리 실패, 순차 처리로 대체: {e}")
                    # 병렬 처리 실패 시 순차 처리로 대체
                    result = self.parallel_processor.process_chunks_sequential(
                        optimized_chunks, 
                        question, 
                        self._single_chunk_rag,
                        early_stop_threshold=0.8
                    )
            else:
                # 단일 청크인 경우 직접 처리
                result = self._single_chunk_rag(optimized_chunks[0].documents, question)
                result = {
                    "answer": result,
                    "chunks_processed": 1,
                    "processing_time": 0.0,
                    "processing_mode": "single_chunk"
                }
            
            # 4. 출처 정보 수집
            all_sources = []
            for chunk in optimized_chunks:
                chunk_sources = self._generate_enhanced_sources(chunk.documents)
                all_sources.extend(chunk_sources)
            
            # 중복 제거
            unique_sources = []
            seen_sources = set()
            for source in all_sources:
                source_key = (source.get('file_name', ''), source.get('chunk_id', ''))
                if source_key not in seen_sources:
                    unique_sources.append(source)
                    seen_sources.add(source_key)
            
            # 5. 검색 정보 수집
            search_info = self._get_search_info(question, relevant_docs)
            search_info.update({
                "chunks_created": len(chunks),
                "chunks_processed": result.get("chunks_processed", len(optimized_chunks)),
                "processing_time": result.get("processing_time", 0.0),
                "chunking_strategy": "hybrid"
            })
            
            # 6. 최종 결과 구성
            return {
                "answer": result.get("answer", "처리 중 오류가 발생했습니다."),
                "sources": unique_sources,
                "status": "success",
                "search_info": search_info,
                "context_tokens": sum(chunk.total_length for chunk in optimized_chunks) // 4,
                "processing_mode": "large_context",
                "chunks_summary": self.context_chunker.get_chunk_summary(optimized_chunks)
            }
            
        except Exception as e:
            logger.error(f"대량 문서 처리 중 오류: {e}")
            # 오류 발생 시 기존 방식으로 대체
            logger.info("기존 방식으로 대체 처리")
            return self._process_standard_context(question, relevant_docs)
    
    async def _run_async_processing(self, chunks: List[ContextChunk], question: str) -> Dict[str, Any]:
        """비동기 처리 실행"""
        return await self.parallel_processor.process_chunks_parallel(
            chunks,
            question,
            self._single_chunk_rag
        )
    
    def _single_chunk_rag(self, documents: List[tuple], question: str) -> str:
        """단일 청크에 대한 RAG 처리"""
        try:
            # 컨텍스트 생성
            context = self._format_documents(documents, question)
            
            # LLM 호출
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            
            # 응답 텍스트 추출
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict) and 'text' in response:
                return response['text']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"단일 청크 RAG 처리 오류: {e}")
            return f"이 문서 청크 처리 중 오류가 발생했습니다: {str(e)}"
    
    def get_large_context_settings(self) -> Dict[str, Any]:
        """대량 컨텍스트 처리 설정 정보 반환"""
        return {
            "enable_large_context_processing": self.enable_large_context_processing,
            "large_context_threshold": self.large_context_threshold,
            "max_chunk_size": self.context_chunker.max_chunk_size,
            "parallel_workers": self.parallel_processor.max_workers,
            "chunk_timeout": self.parallel_processor.chunk_timeout,
            "enable_result_merging": self.parallel_processor.enable_result_merging
        }
    
    def update_large_context_settings(self, **kwargs):
        """대량 컨텍스트 처리 설정 업데이트"""
        if 'enable_large_context_processing' in kwargs:
            self.enable_large_context_processing = kwargs['enable_large_context_processing']
        
        if 'large_context_threshold' in kwargs:
            self.large_context_threshold = kwargs['large_context_threshold']
        
        if 'max_chunk_size' in kwargs:
            self.context_chunker.max_chunk_size = kwargs['max_chunk_size']
        
        if 'parallel_workers' in kwargs:
            self.parallel_processor.max_workers = kwargs['parallel_workers']
        
        if 'chunk_timeout' in kwargs:
            self.parallel_processor.chunk_timeout = kwargs['chunk_timeout']
        
        if 'enable_result_merging' in kwargs:
            self.parallel_processor.enable_result_merging = kwargs['enable_result_merging']
        
        logger.info(f"대량 컨텍스트 처리 설정 업데이트: {kwargs}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """컨텍스트 분할 및 병렬 처리 통계 반환"""
        return {
            "parallel_processing_stats": self.parallel_processor.get_processing_stats(),
            "current_settings": self.get_large_context_settings()
        }
