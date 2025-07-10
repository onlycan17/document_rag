from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from config import settings
from src.vectorstore import VectorDatabase
from src.models import ModelRegistry
from src.utils import TextProcessor
from src.constants import (
    MODEL_PROMPT_TOKENS, MODEL_MIN_CONTEXT, MODEL_MAX_CONTEXT,
    MODEL_SAFETY_MARGIN, LOCAL_MODEL_SAFETY_MARGIN, TOKEN_TO_CHAR_RATIO,
    OPTIMAL_DOC_LENGTH_RANGE, MAX_DOC_LENGTH_SCORE, LOCAL_MODEL_MAX_DOCUMENTS,
    REQUEST_TIMEOUT
)
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, vector_db: Optional[VectorDatabase] = None):
        # 벡터 DB 인스턴스를 외부에서 주입받거나 새로 생성
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDatabase()
        
        self.llm = self._initialize_llm(provider, model)
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
        self.current_provider = provider or settings.llm_provider
        self.current_model = model
        
        # 모델 레지스트리에서 설정 가져오기
        if settings.llm_provider == "local":
            ModelRegistry.update_local_model_config(
                settings.local_llm_max_tokens,
                settings.local_llm_context_window
            )
    
    def _initialize_llm(self, provider: Optional[str] = None, model: Optional[str] = None):
        """LLM 초기화"""
        # provider가 지정되지 않으면 설정에서 가져옴
        if provider is None:
            provider = settings.llm_provider
        
        # 실제 사용할 모델명 결정
        actual_model = model or getattr(settings, f"{provider}_model", None)
        
        # 모델별 최대 토큰 수 가져오기
        max_tokens = self._get_max_tokens_for_model(provider, actual_model) if actual_model else settings.max_tokens
        
        logger.info(f"LLM 초기화: provider={provider}, model={actual_model}, max_tokens={max_tokens}")
            
        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            return ChatOpenAI(
                openai_api_key=settings.openai_api_key,
                model_name=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "google":
            if not settings.google_api_key:
                raise ValueError("Google API 키가 설정되지 않았습니다.")
            return ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=actual_model,
                temperature=settings.temperature,
                max_output_tokens=max_tokens
            )
        
        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
            return ChatAnthropic(
                anthropic_api_key=settings.anthropic_api_key,
                model_name=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens
            )
        
        elif provider == "local":
            # OpenAI 호환 API를 사용하는 로컬 모델
            return ChatOpenAI(
                openai_api_key=settings.local_llm_api_key,
                openai_api_base=settings.local_llm_base_url + "/v1",
                model_name=actual_model,
                temperature=settings.temperature,
                max_tokens=max_tokens
            )
        
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        template = """당신은 전문적인 지식을 친절하게 전달하는 AI 어시스턴트입니다. 
상세하고 정확한 정보를 제공하면서도, 일반인이 이해할 수 있도록 설명하는 것이 당신의 역할입니다.

다음은 검색된 관련 문서들입니다:
{context}

위 문서들의 내용을 참고하여 다음 질문에 답변해주세요:
{question}

답변 시 반드시 지켜야 할 규칙:

1. **상세하고 풍부한 정보 제공**:
   - 문서에 있는 구체적인 사실, 숫자, 날짜, 발견사항을 빠짐없이 포함하세요
   - 중요한 세부사항을 생략하지 말고 충실히 전달하세요
   - 관련된 모든 정보를 체계적으로 정리하여 상세하게 제공하세요

2. **이해하기 쉬운 설명**:
   - 전문 용어는 처음 나올 때 괄호 안에 쉬운 설명을 추가하세요
   - 중요한 내용은 강조하여 설명하세요
   - 어려운 용어 및 개념은 예시를 통해 설명하세요
   - 복잡한 내용은 단계별로 나누어 설명하세요
   - 필요한 경우 배경 지식을 상세히 제공하세요

3. **구조화된 답변**:
   - 주제별로 섹션을 나누어 정리하세요
   - 시대순, 중요도순 등 논리적인 순서로 배열하세요
   - 번호나 불릿 포인트를 활용하여 가독성을 높이세요

4. **정확성과 신뢰성**:
   - 제공된 문서의 내용을 정확히 인용하세요
   - 추측이나 일반화는 피하고, 문서에 근거한 사실만 전달하되 상세히 전달하세요
   - 출처가 명확한 정보는 출처를 함께 언급하세요

5. **종합적인 답변**:
   - 질문의 핵심에 대한 직접적인 답변을 먼저 제공하세요
   - 그 다음 관련된 부가 정보와 맥락을 상세히 설명하세요
   - 가능한 한 많은 관련 정보를 포함하여 완전한 답변을 만드세요

답변:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _create_chain(self):
        """LLM 체인 생성"""
        # 새로운 LCEL 방식 사용
        return self.prompt_template | self.llm
    
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
        if provider == "local":
            max_context = MODEL_MAX_CONTEXT["local"]
        elif provider == "openai" and model and "gpt-3.5" in model:
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
            
            # 5. 종합 점수 계산
            final_score = (
                relevance_score * 0.5 +
                quality_score * 0.2 +
                length_score * 0.2 +
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
        professional_terms = ['시스템', '정보', '구축', '운영', '관리', '지침', '규정', '법령']
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
        """컨텐츠 최적화 - 불필요한 공백만 정리"""
        # 불필요한 공백 정리
        content = ' '.join(content.split())
        
        # 길이 제한 없이 전체 내용 반환
        return content
    
    def get_last_context_tokens(self) -> int:
        """마지막 쿼리에서 사용된 컨텍스트 토큰 수 반환"""
        return getattr(self, '_last_context_tokens', 0)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        향상된 질문 처리
        - 쿼리 전처리 및 확장
        - 검색 결과 분석
        - 컨텍스트 최적화
        """
        try:
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
            
            relevant_docs = self.vector_db.search(processed_question, k=k_docs)
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
                "context_tokens": self._last_context_tokens
            }
            
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류 발생: {str(e)}")
            return {
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류 정보: {str(e)}",
                "sources": [],
                "status": "error",
                "search_info": {}
            }
    
    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리 및 확장"""
        if not settings.enable_query_preprocessing:
            return query
        
        # 기본 정제
        processed_query = query.strip()
        
        # 동적 쿼리 확장 - 문서 기반
        if settings.enable_query_expansion:
            # 먼저 문서에서 관련 용어 추출
            related_terms = self.vector_db.extract_related_terms(query)
            if related_terms:
                logger.info(f"문서 기반 관련 용어 추출: {related_terms[:10]}")
                
                # 기존 쿼리에 관련 용어 추가
                all_terms = query.split() + related_terms[:10]  # 상위 10개만 사용
                
                # 중복 제거 (순서 유지)
                seen = set()
                unique_terms = []
                for term in all_terms:
                    if term not in seen and len(term) > 1:
                        seen.add(term)
                        unique_terms.append(term)
                
                processed_query = ' '.join(unique_terms[:20])  # 최대 20개 단어
                logger.info(f"동적 쿼리 확장 완료: '{query}' -> '{processed_query}'")
                return processed_query
        
        # 동적 확장이 실패하거나 비활성화된 경우, 정적 확장 사용
        if settings.enable_query_expansion:
            processed_query = TextProcessor.expand_query(processed_query, use_static_expansion=True)
        
        logger.info(f"쿼리 전처리: '{query}' -> '{processed_query}'")
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
                "chunk_info": f"{doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}"
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
        self.llm = self._initialize_llm(provider, model)
        self.chain = self._create_chain()
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
        
        # 로컬 모델이 실행 중인 경우, 실제 모델 목록 가져오기
        local_models = self._get_local_models()
        if local_models:
            models["local"] = local_models
        
        return models
    
    def _get_local_models(self) -> List[Dict[str, str]]:
        """로컬 서버에서 사용 가능한 모델 목록 조회"""
        try:
            import requests  # 로컬 import로 옵셔널 의존성 처리
            response = requests.get(f"{settings.local_llm_base_url}/v1/models", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                local_models = []
                for model in data.get("data", []):
                    local_models.append({
                        "name": model.get("id", "unknown"),
                        "model": model.get("id", "unknown"),
                        "description": f"로컬 모델 - {model.get('owned_by', 'unknown')}"
                    })
                return local_models if local_models else [{"name": "로컬 모델", "model": "local-model", "description": "현재 실행 중인 모델"}]
        except:
            pass
        return []
    
    def add_feedback(self, question: str, answer: str, feedback: str):
        """사용자 피드백 저장 (향후 개선을 위한 기능)"""
        # TODO: 피드백을 데이터베이스에 저장하여 모델 개선에 활용
        pass