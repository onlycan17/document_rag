"""
리팩터링된 RAG 체인 모듈

이 모듈은 기존의 monolithic RAG 체인을 모듈화된 구조로 리팩터링한 메인 오케스트레이터입니다.
LLMManager, DocumentProcessor, QueryEngine 모듈들을 조합하여 전체 RAG 파이프라인을 관리합니다.
"""

from typing import List, Dict, Any, Optional, Generator, Tuple
from langchain.schema import Document
import logging

from config import settings
from src.vectorstore import VectorDatabase

# 새로 생성한 모듈들 임포트
from .llm_manager import LLMManager
from .document_processor import DocumentProcessor
from .query_engine import QueryEngine
from .context_chunker import ContextChunker
from .summarizer import HierarchicalSummarizer, resolve_summary_length
from .rag_parallel_processor import ParallelRAGProcessor

logger = logging.getLogger(__name__)


class RAGChain:
    """
    리팩터링된 RAG 체인 클래스

    기존 RAG 체인의 모든 기능을 유지하면서 모듈화된 구조를 사용합니다.
    - LLMManager: LLM 초기화 및 관리
    - DocumentProcessor: 문서 처리 및 최적화
    - QueryEngine: 쿼리 처리 및 응답 생성
    """

    def __init__(
        self, provider: Optional[str] = None, model: Optional[str] = None, vector_db: Optional[VectorDatabase] = None
    ):
        """
        RAG 체인 초기화

        Args:
            provider: LLM 제공자 (openai, google, anthropic, local)
            model: 사용할 모델명
            vector_db: 벡터 데이터베이스 인스턴스
        """
        # 벡터 DB 인스턴스를 외부에서 주입받거나 새로 생성
        if vector_db is not None:
            self.vector_db = vector_db
        else:
            self.vector_db = VectorDatabase()

        # 현재 설정 저장
        self.current_provider = provider or settings.llm_provider
        self.current_model = model

        # 모듈화된 구성 요소들 초기화
        self.llm_manager = LLMManager()
        self.document_processor = DocumentProcessor(max_context_length_func=self._get_max_context_length_for_model)
        self.query_engine = QueryEngine(
            vector_db=self.vector_db, llm_manager=self.llm_manager, document_processor=self.document_processor
        )

        # LLM 및 체인 초기화
        self._initialize_components(provider, model)

        # 기존 고급 기능들 초기화 (있는 경우에만)
        self._initialize_advanced_features()

        logger.info(f"RAG 체인 초기화 완료 - Provider: {self.current_provider}, Model: {self.current_model}")

    def _initialize_components(self, provider: Optional[str], model: Optional[str]):
        """모든 구성 요소 초기화"""
        # LLM 및 체인 생성
        self.llm = self.llm_manager.create_llm(provider, model, streaming=False)
        self.streaming_llm = self.llm_manager.create_llm(provider, model, streaming=True)
        self.prompt_template = self.llm_manager.create_prompt_template()

        # 체인 생성
        self.chain = self.llm_manager.create_chain(self.llm, self.prompt_template)
        self.streaming_chain = self.llm_manager.create_chain(self.streaming_llm, self.prompt_template)

        # QueryEngine에 체인 설정
        self.query_engine.set_chains(self.chain, self.streaming_chain)

    def _initialize_advanced_features(self):
        """고급 기능들 초기화"""
        self.context_chunker = ContextChunker(max_chunk_size=30000)  # 30KB 청크
        summary_length = resolve_summary_length(
            self.llm_manager.get_max_tokens_for_model(self.current_provider, self.current_model)
        )
        self.summarizer = HierarchicalSummarizer(llm=self.llm, max_summary_length=summary_length)
        logger.info(f"요약 목표 길이 자동 튜닝: {summary_length}자 (모델: {self.current_model})")
        self.parallel_processor = ParallelRAGProcessor(max_workers=3, chunk_timeout=30.0, enable_result_merging=True)
        self.parallel_processor.set_summarizer(self.summarizer)

        # 대량 문서 처리 활성화 여부
        self.enable_large_context_processing = getattr(settings, "enable_large_context_processing", True)
        self.large_context_threshold = getattr(settings, "large_context_threshold", 50000)  # 50KB 임계값

    def _get_max_context_length_for_model(self) -> int:
        """현재 모델의 최대 컨텍스트 길이 반환"""
        return self.llm_manager.get_max_context_length_for_model(self.current_provider, self.current_model)

    def query(self, question: str, k_documents: int = 8) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성 (비스트리밍)

        Args:
            question: 사용자 질문
            k_documents: 검색할 문서 수 (현재는 설정값 사용)

        Returns:
            답변 정보가 포함된 딕셔너리
        """
        # k_documents 파라미터는 현재 설정값을 사용하므로 무시됨
        return self.query_engine.query(question)

    def invoke(self, question: str) -> Dict[str, Any]:
        """
        LangChain Runnable 인터페이스 호환성을 위한 invoke 메서드

        Args:
            question: 사용자 질문 (문자열 또는 딕셔너리)

        Returns:
            답변 정보가 포함된 딕셔너리
        """
        # 입력이 딕셔너리인 경우 질문 추출
        if isinstance(question, dict):
            question = question.get("question", question.get("input", str(question)))

        return self.query(question)

    def stream_query(self, question: str, k_documents: int = 8) -> Generator[Dict[str, Any], None, None]:
        """
        질문에 대한 답변을 스트리밍으로 생성

        Args:
            question: 사용자 질문
            k_documents: 검색할 문서 수 (현재는 설정값 사용)

        Yields:
            스트리밍 응답 청크
        """
        # k_documents 파라미터는 현재 설정값을 사용하므로 무시됨
        yield from self.query_engine.stream_query(question)

    def update_llm(self, provider: str, model: Optional[str] = None):
        """
        LLM 업데이트

        Args:
            provider: 새로운 LLM 제공자
            model: 새로운 모델명
        """
        logger.info(f"LLM 업데이트: {provider} / {model}")

        # 현재 설정 업데이트
        self.current_provider = provider
        self.current_model = model

        # 구성 요소들 재초기화
        self._initialize_components(provider, model)

        # 고급 기능들도 새 LLM으로 업데이트
        if self.summarizer and self.llm:
            self.summarizer.llm = self.llm
            self.summarizer.max_summary_length = resolve_summary_length(
                self.llm_manager.get_max_tokens_for_model(provider, model)
            )

    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """사용 가능한 모델 목록 반환"""
        return self.llm_manager.get_available_models()

    def get_current_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        models = self.get_available_models()
        provider_models = models.get(self.current_provider, [])

        for model_info in provider_models:
            if model_info.get("name") == self.current_model:
                return model_info

        return {
            "provider": self.current_provider,
            "name": self.current_model,
            "description": "현재 사용 중인 모델",
            "max_tokens": "알 수 없음",
            "context_window": "알 수 없음",
        }

    def search_documents(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        """
        벡터 DB에서 문서 검색

        Args:
            query: 검색 쿼리
            k: 검색할 문서 수

        Returns:
            (Document, score) 튜플 리스트
        """
        return self.query_engine.search_documents(query, k)

    def format_documents(self, documents: List[Tuple[Document, float]], query: str = "") -> str:
        """
        검색된 문서들을 컨텍스트로 포맷

        Args:
            documents: (Document, score) 튜플 리스트
            query: 검색 쿼리

        Returns:
            포맷된 컨텍스트 문자열
        """
        return self.document_processor.format_documents(documents, query)

    def generate_enhanced_sources(self, documents: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """향상된 출처 정보 생성"""
        return self.document_processor.generate_enhanced_sources(documents)

    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        return self.query_engine.preprocess_query(query)

    def get_stats(self) -> Dict[str, Any]:
        """RAG 체인 통계 정보 반환"""
        return {
            "current_provider": self.current_provider,
            "current_model": self.current_model,
            "vector_db_type": settings.vector_db_type,
            "vector_db_status": "연결됨" if self.vector_db else "연결 안됨",
            "llm_status": "초기화됨" if self.llm else "초기화 안됨",
            "streaming_llm_status": "초기화됨" if self.streaming_llm else "초기화 안됨",
            "advanced_features": {
                "context_chunker": self.context_chunker is not None,
                "summarizer": self.summarizer is not None,
                "parallel_processor": self.parallel_processor is not None,
                "large_context_processing": getattr(self, "enable_large_context_processing", False),
            },
        }

    # 기존 API 호환성을 위한 메서드들
    def _get_model_max_tokens(self) -> Dict[str, int]:
        """모델별 최대 토큰 수 반환 (기존 호환성)"""
        return self.llm_manager._get_model_max_tokens()

    def _get_model_context_window(self) -> Dict[str, int]:
        """모델별 컨텍스트 윈도우 크기 반환 (기존 호환성)"""
        return self.llm_manager._get_model_context_window()

    def _get_max_tokens_for_model(self, provider: str, model: Optional[str] = None) -> int:
        """특정 모델의 최대 토큰 수 반환 (기존 호환성)"""
        return self.llm_manager._get_max_tokens_for_model(provider, model)

    # 대량 문서 처리 메서드들 (기존 기능 유지)
    def process_large_context(self, question: str, k_documents: int = 15) -> Dict[str, Any]:
        """대량 컨텍스트 처리 (기존 기능)"""
        if not self.enable_large_context_processing or not self.parallel_processor:
            logger.warning("대량 컨텍스트 처리가 비활성화되어 있습니다. 일반 처리로 진행합니다.")
            return self.query(question, k_documents)

        try:
            # 문서 검색
            documents = self.search_documents(question, k_documents)
            context = self.format_documents(documents, question)

            # 대량 컨텍스트인지 확인
            if len(context) < self.large_context_threshold:
                logger.info("일반 크기 컨텍스트로 처리")
                return self.query(question, k_documents)

            logger.info(f"대량 컨텍스트 처리 시작: {len(context):,}자")

            # 컨텍스트 분할
            chunks = self.context_chunker.split_context(context)

            # 병렬 처리
            results = self.parallel_processor.process_chunks(chunks, question)

            # 결과 병합
            if results:
                final_answer = self.parallel_processor.merge_results(results, question)
                sources = self.generate_enhanced_sources(documents)

                return {
                    "answer": final_answer,
                    "sources": sources,
                    "context_length": len(context),
                    "chunks_processed": len(chunks),
                    "processing_method": "parallel_large_context",
                }
            else:
                logger.warning("병렬 처리 실패, 일반 처리로 폴백")
                return self.query(question, k_documents)

        except Exception as e:
            logger.error(f"대량 컨텍스트 처리 오류 ({type(e).__name__}): {e}", exc_info=True)
            logger.info("일반 처리로 폴백")
            return self.query(question, k_documents)

    def stream_large_context(self, question: str, k_documents: int = 15) -> Generator[Dict[str, Any], None, None]:
        """대량 컨텍스트 스트리밍 처리 (기존 기능)"""
        if not self.enable_large_context_processing or not self.parallel_processor:
            logger.warning("대량 컨텍스트 처리가 비활성화되어 있습니다. 일반 스트리밍으로 진행합니다.")
            yield from self.stream_query(question, k_documents)
            return

        try:
            # 문서 검색
            documents = self.search_documents(question, k_documents)
            context = self.format_documents(documents, question)

            # 대량 컨텍스트인지 확인
            if len(context) < self.large_context_threshold:
                logger.info("일반 크기 컨텍스트로 스트리밍")
                yield from self.stream_query(question, k_documents)
                return

            logger.info(f"대량 컨텍스트 스트리밍 시작: {len(context):,}자")

            # 일반 스트리밍으로 폴백 (대량 컨텍스트 스트리밍은 복잡)
            yield from self.stream_query(question, k_documents)

        except Exception as e:
            logger.error(f"대량 컨텍스트 스트리밍 오류 ({type(e).__name__}): {e}", exc_info=True)
            yield from self.stream_query(question, k_documents)
