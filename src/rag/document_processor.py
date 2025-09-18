"""
문서 처리 모듈

이 모듈은 RAG 시스템에서 검색된 문서들의 처리를 담당합니다.
중복 제거, 재랭킹, 품질 평가, 컨텐츠 최적화 등의 기능을 제공합니다.
"""

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import re
import logging

from config import settings
from src.constants import (
    OPTIMAL_DOC_LENGTH_RANGE, MAX_DOC_LENGTH_SCORE
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    문서 처리를 담당하는 클래스
    
    주요 기능:
    - 문서 포맷팅 및 컨텍스트 생성
    - 중복 문서 제거
    - 문서 재랭킹 (관련성, 품질, 다양성 고려)
    - 컨텐츠 최적화 및 정제
    """
    
    def __init__(self, max_context_length_func=None):
        """
        문서 처리기 초기화
        
        Args:
            max_context_length_func: 최대 컨텍스트 길이를 반환하는 함수
        """
        self.get_max_context_length = max_context_length_func or self._default_max_context_length
    
    def _default_max_context_length(self) -> int:
        """기본 최대 컨텍스트 길이"""
        return 100000  # 100KB
    
    def format_documents(self, documents: List[tuple], query: str = "") -> str:
        """
        검색된 문서를 컨텍스트로 포맷
        - 중복 제거 및 재랭킹
        - 컨텍스트 최적화
        - 관련성 점수 기반 정렬
        
        Args:
            documents: (Document, score) 튜플 리스트
            query: 검색 쿼리
            
        Returns:
            포맷된 컨텍스트 문자열
        """
        if not documents:
            return ""
        
        # 1. 중복 문서 제거 (유사한 내용 필터링)
        unique_documents = self.remove_duplicate_documents(documents)
        
        # 2. 재랭킹 (관련성과 다양성 모두 고려)
        reranked_documents = self.rerank_documents(unique_documents)
        
        # 3. 컨텍스트 최적화
        context_parts = []
        total_length = 0
        max_context_length = self.get_max_context_length()  # 모델별 동적 컨텍스트 길이
        
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
            content = self.optimize_content(doc.page_content.strip(), query)
            
            # 컨텍스트 길이 체크
            content_preview = f"[문서 {i+1}: {file_name} (관련도: {relevance_percent:.1f}%, {chunk_info})]\n{content}\n"
            
            if total_length + len(content_preview) > max_context_length:
                logger.info(f"컨텍스트 길이 제한으로 {i+1}번째 문서에서 중단")
                break
            
            context_parts.append(content_preview)
            total_length += len(content_preview)
        
        final_context = "\n" + "="*50 + "\n".join(context_parts) + "="*50 + "\n"
        
        logger.info(f"컨텍스트 구성 완료: {len(context_parts)}개 문서, {total_length}자")
        return final_context
    
    def remove_duplicate_documents(self, documents: List[tuple]) -> List[tuple]:
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
                if self.calculate_content_similarity(content, str(existing_content)) > 0.9:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_docs.append((doc, score))
                seen_contents.add(content_hash)
        
        logger.info(f"중복 제거: {len(documents)} -> {len(unique_docs)}개 문서")
        return unique_docs
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
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
    
    def rerank_documents(self, documents: List[tuple]) -> List[tuple]:
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
            relevance_score = self.normalize_score(original_score)
            
            # 2. 문서 품질 점수
            quality_score = self.calculate_document_quality(doc)
            
            # 3. 컨텐츠 길이 점수 (너무 짧거나 긴 문서 패널티)
            length_score = self.calculate_length_score(doc.page_content)
            
            # 4. 메타데이터 품질 점수
            metadata_score = self.calculate_metadata_quality(doc.metadata)
            
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
    
    def normalize_score(self, score: float) -> float:
        """점수 정규화 (0-1 범위로)"""
        if settings.vector_db_type == "faiss":
            # FAISS 거리를 유사도로 변환
            return max(0, min(1, (2.0 - float(score)) / 2.0))
        else:
            # ChromaDB 유사도 그대로 사용
            return min(1, max(0, float(score)))
    
    def calculate_document_quality(self, doc: Document) -> float:
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
    
    def calculate_length_score(self, content: str) -> float:
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
    
    def calculate_metadata_quality(self, metadata: dict) -> float:
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
    
    def optimize_content(self, content: str, query: str = "") -> str:
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

    def sanitize_output_chunk(self, text: str) -> str:
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
    
    def generate_enhanced_sources(self, documents: List[tuple]) -> List[Dict[str, Any]]:
        """향상된 출처 정보 생성"""
        sources = []
        
        for i, (doc, score) in enumerate(documents):
            # 관련성 점수 표시 형태 개선
            if settings.vector_db_type == "faiss":
                # FAISS는 거리 기반 (낮을수록 좋음)
                relevance_percent = max(0, min(100, (2.0 - float(score)) * 50))
            else:
                # ChromaDB는 유사도 기반 (높을수록 좋음)
                relevance_percent = min(100, float(score) * 100)
            
            source_info = {
                "index": i + 1,
                "source": doc.metadata.get('source', '알 수 없음'),
                "file_name": doc.metadata.get('file_name', '알 수 없음'),
                "page": doc.metadata.get('page', 'N/A'),
                "chunk_id": doc.metadata.get('chunk_id', f'chunk_{i}'),
                "relevance_score": relevance_percent,
                "content_length": len(doc.page_content),
                "extraction_method": doc.metadata.get('extraction_method', '알 수 없음'),
                "processing_method": doc.metadata.get('processing_method', '알 수 없음'),
                "document_quality": self.calculate_document_quality(doc),
                "metadata_quality": self.calculate_metadata_quality(doc.metadata)
            }
            
            sources.append(source_info)
        
        return sources