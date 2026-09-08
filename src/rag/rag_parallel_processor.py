"""
병렬 RAG 처리 시스템

여러 컨텍스트 청크를 동시에 처리하여 대량 문서에서
빠르고 효율적인 답변을 생성하는 시스템입니다.
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from langchain.schema import Document
from dataclasses import dataclass
import logging
import time
from threading import Lock

from .context_chunker import ContextChunk
from .summarizer import HierarchicalSummarizer

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """처리 결과를 담는 클래스"""

    chunk_id: str
    success: bool
    result: Optional[str]
    metadata: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None


class ParallelRAGProcessor:
    """병렬 RAG 처리 관리 클래스"""

    def __init__(self, max_workers: int = 3, chunk_timeout: float = 30.0, enable_result_merging: bool = True):
        """
        Args:
            max_workers: 동시 처리할 최대 작업자 수
            chunk_timeout: 각 청크 처리 타임아웃 (초)
            enable_result_merging: 결과 병합 활성화 여부
        """
        self.max_workers = max_workers
        self.chunk_timeout = chunk_timeout
        self.enable_result_merging = enable_result_merging
        self.processing_lock = Lock()
        self.summarizer = None

        # 성능 통계
        self.processing_stats = {
            "total_chunks_processed": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "total_processing_time": 0.0,
            "average_chunk_time": 0.0,
        }

    def set_summarizer(self, summarizer: HierarchicalSummarizer):
        """요약 엔진 설정"""
        self.summarizer = summarizer

    async def process_chunks_parallel(
        self, chunks: List[ContextChunk], query: str, rag_chain_func: Callable
    ) -> Dict[str, Any]:
        """
        병렬로 청크들을 처리

        Args:
            chunks: 처리할 컨텍스트 청크들
            query: 사용자 쿼리
            rag_chain_func: RAG 체인 함수 (단일 청크 처리용)

        Returns:
            병합된 최종 결과
        """
        if not chunks:
            return {"answer": "", "chunks_processed": 0, "processing_time": 0.0}

        logger.info(f"병렬 처리 시작: {len(chunks)}개 청크, 최대 {self.max_workers}개 작업자")
        start_time = time.time()

        # 청크 중요도에 따른 우선순위 설정
        prioritized_chunks = self._prioritize_chunks(chunks, query)

        # 비동기 처리 실행
        processing_results = await self._process_chunks_async(prioritized_chunks, query, rag_chain_func)

        # 결과 병합
        if self.enable_result_merging:
            final_result = await self._merge_results(processing_results, query)
        else:
            # 가장 좋은 결과만 선택
            best_result = self._select_best_result(processing_results)
            final_result = {
                "answer": best_result.result if best_result else "",
                "source_chunk": best_result.chunk_id if best_result else None,
            }

        # 처리 통계 업데이트
        total_time = time.time() - start_time
        self._update_processing_stats(processing_results, total_time)

        # 최종 결과에 메타데이터 추가
        final_result.update(
            {
                "chunks_processed": len(processing_results),
                "successful_chunks": sum(1 for r in processing_results if r.success),
                "total_processing_time": total_time,
                "processing_stats": self.processing_stats.copy(),
            }
        )

        logger.info(
            f"병렬 처리 완료: {total_time:.2f}초, "
            f"{len([r for r in processing_results if r.success])}/{len(processing_results)} 성공"
        )

        return final_result

    def process_chunks_sequential(
        self, chunks: List[ContextChunk], query: str, rag_chain_func: Callable, early_stop_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        순차적으로 청크들을 처리 (조기 종료 지원)

        Args:
            chunks: 처리할 컨텍스트 청크들
            query: 사용자 쿼리
            rag_chain_func: RAG 체인 함수
            early_stop_threshold: 조기 종료 임계값 (신뢰도)
        """
        if not chunks:
            return {"answer": "", "chunks_processed": 0, "processing_time": 0.0}

        logger.info(f"순차 처리 시작: {len(chunks)}개 청크")
        start_time = time.time()

        results = []
        processed_count = 0

        # 중요도 순으로 정렬
        prioritized_chunks = self._prioritize_chunks(chunks, query)

        for chunk in prioritized_chunks:
            try:
                # 단일 청크 처리
                chunk_result = self._process_single_chunk(chunk, query, rag_chain_func)
                results.append(chunk_result)
                processed_count += 1

                # 조기 종료 조건 확인
                if chunk_result.success and self._should_early_stop(chunk_result, early_stop_threshold):
                    logger.info(f"조기 종료: 신뢰도 {early_stop_threshold} 이상 달성")
                    break

            except Exception as e:
                logger.error(f"청크 {chunk.chunk_id} 처리 중 오류: {e}")
                results.append(
                    ProcessingResult(
                        chunk_id=chunk.chunk_id,
                        success=False,
                        result=None,
                        metadata={},
                        processing_time=0.0,
                        error_message=str(e),
                    )
                )

        # 최종 결과 생성
        if self.enable_result_merging and len(results) > 1:
            # 여러 결과 병합
            final_answer = self._merge_sequential_results(results, query)
        else:
            # 가장 좋은 결과 선택
            best_result = self._select_best_result(results)
            final_answer = best_result.result if best_result else ""

        total_time = time.time() - start_time

        return {
            "answer": final_answer,
            "chunks_processed": processed_count,
            "total_chunks": len(chunks),
            "processing_time": total_time,
            "early_stopped": processed_count < len(chunks),
        }

    async def _process_chunks_async(
        self, chunks: List[ContextChunk], query: str, rag_chain_func: Callable
    ) -> List[ProcessingResult]:
        """비동기로 청크들을 처리"""
        # ThreadPoolExecutor를 사용하여 동기 함수를 비동기로 실행
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 청크에 대한 태스크 생성
            tasks = []
            for chunk in chunks:
                task = loop.run_in_executor(executor, self._process_single_chunk, chunk, query, rag_chain_func)
                tasks.append(task)

            # 모든 태스크 완료 대기 (타임아웃 적용)
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=self.chunk_timeout * len(chunks)
                )

                # 예외 처리
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        processed_results.append(
                            ProcessingResult(
                                chunk_id=chunks[i].chunk_id,
                                success=False,
                                result=None,
                                metadata={},
                                processing_time=0.0,
                                error_message=str(result),
                            )
                        )
                    else:
                        processed_results.append(result)

                return processed_results

            except asyncio.TimeoutError:
                logger.warning(f"청크 처리 타임아웃: {self.chunk_timeout * len(chunks)}초")
                # 타임아웃 발생 시 빈 결과 반환
                return [
                    ProcessingResult(
                        chunk_id=chunk.chunk_id,
                        success=False,
                        result=None,
                        metadata={},
                        processing_time=0.0,
                        error_message="처리 타임아웃",
                    )
                    for chunk in chunks
                ]

    def _process_single_chunk(self, chunk: ContextChunk, query: str, rag_chain_func: Callable) -> ProcessingResult:
        """단일 청크 처리 (재시도 로직 포함)"""
        start_time = time.time()

        max_retries = 1

        last_error = None

        for attempt in range(max_retries):
            try:
                # 재시도 시 로깅

                # 청크 내용을 문서 형태로 변환
                chunk_documents = chunk.documents

                result = rag_chain_func(chunk_documents, query)

                processing_time = time.time() - start_time

                # 성공 시 결과 반환
                return ProcessingResult(
                    chunk_id=chunk.chunk_id,
                    success=True,
                    result=result,
                    metadata={
                        "chunk_size": chunk.total_length,
                        "document_count": len(chunk.documents),
                        "relevance_score": chunk.relevance_score,
                        "topic_keywords": chunk.topic_keywords,
                        "attempts": attempt + 1,  # 시도 횟수 기록
                    },
                    processing_time=processing_time,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"청크 {chunk.chunk_id} 처리 실패 (시도 {attempt+1}/{max_retries}): {str(e)}")

                # 마지막 시도가 아니면 계속 시도
                if attempt < max_retries - 1:
                    continue

        # 모든 재시도 실패 시
        processing_time = time.time() - start_time
        logger.error(f"청크 {chunk.chunk_id} 최종 처리 실패: {last_error}")

        return ProcessingResult(
            chunk_id=chunk.chunk_id,
            success=False,
            result=None,
            metadata={"attempts": max_retries},
            processing_time=processing_time,
            error_message=str(last_error),
        )

    def _prioritize_chunks(self, chunks: List[ContextChunk], query: str) -> List[ContextChunk]:
        """청크 우선순위 설정"""

        def calculate_priority_score(chunk: ContextChunk) -> float:
            """우선순위 점수 계산"""
            # 기본 관련도 점수
            base_score = chunk.relevance_score

            # 쿼리와의 키워드 매칭 점수
            query_words = set(query.lower().split())
            chunk_text = chunk.get_content().lower()
            chunk_words = set(chunk_text.split())

            keyword_overlap = len(query_words.intersection(chunk_words))
            keyword_score = keyword_overlap / len(query_words) if query_words else 0

            # 문서 크기 점수 (적당한 크기가 좋음)
            optimal_size = 10000  # 최적 크기
            size_score = min(1.0, optimal_size / max(chunk.total_length, 1))

            # 최종 우선순위 점수
            priority_score = base_score * 0.5 + keyword_score * 0.3 + size_score * 0.2

            return priority_score

        # 우선순위 점수에 따라 정렬
        return sorted(chunks, key=calculate_priority_score, reverse=True)

    async def _merge_results(self, results: List[ProcessingResult], query: str) -> Dict[str, Any]:
        """여러 처리 결과를 병합"""
        successful_results = [r for r in results if r.success and r.result]

        if not successful_results:
            return {"answer": "죄송합니다. 관련 정보를 찾을 수 없습니다.", "confidence": 0.0}

        if len(successful_results) == 1:
            return {
                "answer": successful_results[0].result,
                "confidence": 1.0,
                "source_chunks": [successful_results[0].chunk_id],
            }

        # 여러 결과가 있는 경우 병합 전략 적용
        return await self._advanced_merge_strategy(successful_results, query)

    async def _advanced_merge_strategy(self, results: List[ProcessingResult], query: str) -> Dict[str, Any]:
        """고급 병합 전략"""
        # 결과들을 관련도 순으로 정렬
        sorted_results = sorted(results, key=lambda r: r.metadata.get("relevance_score", 0), reverse=True)

        # 상위 3개 결과만 사용
        top_results = sorted_results[:3]

        # 요약기가 있는 경우 계층적 요약 사용
        if self.summarizer:
            try:
                # 결과들을 문서로 변환
                result_documents = []
                for result in top_results:
                    doc = Document(
                        page_content=result.result,
                        metadata={
                            "chunk_id": result.chunk_id,
                            "relevance_score": result.metadata.get("relevance_score", 0),
                        },
                    )
                    result_documents.append((doc, result.metadata.get("relevance_score", 0)))

                # 계층적 요약 실행
                summary_result = self.summarizer.create_multi_level_summary(result_documents, query, levels=2)

                return {
                    "answer": summary_result["final_summary"],
                    "confidence": 0.9,
                    "source_chunks": [r.chunk_id for r in top_results],
                    "merge_method": "hierarchical_summary",
                }

            except Exception as e:
                logger.warning(f"계층적 요약 실패, 단순 병합 사용: {e}")

        # 단순 병합 전략
        merged_content = self._simple_merge_strategy(top_results, query)

        return {
            "answer": merged_content,
            "confidence": 0.8,
            "source_chunks": [r.chunk_id for r in top_results],
            "merge_method": "simple_merge",
        }

    def _simple_merge_strategy(self, results: List[ProcessingResult], query: str) -> str:
        """단순 병합 전략"""
        if not results:
            return ""

        if len(results) == 1:
            return results[0].result

        # 중복 제거를 위한 핵심 문장 추출
        unique_sentences = set()
        merged_parts = []

        for i, result in enumerate(results):
            content = result.result.strip()
            if not content:
                continue

            # 문장 단위로 분할
            sentences = [s.strip() for s in content.split(".") if s.strip()]

            # 새로운 정보만 추가
            new_sentences = []
            for sentence in sentences:
                if sentence not in unique_sentences and len(sentence) > 20:
                    unique_sentences.add(sentence)
                    new_sentences.append(sentence)

            if new_sentences:
                section_header = f"\n[출처 {i+1}]\n" if len(results) > 1 else ""
                merged_parts.append(section_header + ". ".join(new_sentences) + ".")

        return "\n\n".join(merged_parts)

    def _merge_sequential_results(self, results: List[ProcessingResult], query: str) -> str:
        """순차 처리 결과 병합"""
        successful_results = [r for r in results if r.success and r.result]

        if not successful_results:
            return "죄송합니다. 관련 정보를 찾을 수 없습니다."

        # 가장 좋은 결과를 기본으로 하고, 추가 정보가 있으면 보완
        best_result = max(successful_results, key=lambda r: r.metadata.get("relevance_score", 0))

        if len(successful_results) == 1:
            return best_result.result

        # 추가 정보 병합
        additional_info = []
        for result in successful_results:
            if result.chunk_id != best_result.chunk_id and result.result:
                # 중복되지 않는 새로운 정보만 추가
                if not self._is_content_duplicate(best_result.result, result.result):
                    additional_info.append(result.result)

        if additional_info:
            return best_result.result + "\n\n[추가 정보]\n" + "\n".join(additional_info[:2])
        else:
            return best_result.result

    def _select_best_result(self, results: List[ProcessingResult]) -> Optional[ProcessingResult]:
        """가장 좋은 결과 선택"""
        successful_results = [r for r in results if r.success and r.result]

        if not successful_results:
            return None

        # 관련도 점수가 가장 높은 결과 선택
        return max(successful_results, key=lambda r: r.metadata.get("relevance_score", 0))

    def _should_early_stop(self, result: ProcessingResult, threshold: float) -> bool:
        """조기 종료 여부 판단"""
        if not result.success:
            return False

        # 관련도 점수 기반 판단
        relevance_score = result.metadata.get("relevance_score", 0)

        # 답변 길이 기반 판단 (너무 짧으면 불완전할 가능성)
        answer_length = len(result.result) if result.result else 0
        min_answer_length = 100

        return relevance_score >= threshold and answer_length >= min_answer_length

    def _is_content_duplicate(self, content1: str, content2: str) -> bool:
        """컨텐츠 중복 여부 확인"""
        if not content1 or not content2:
            return False

        # 간단한 중복 검사 (Jaccard 유사도 사용)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard_similarity = intersection / union if union > 0 else 0

        return jaccard_similarity > 0.7  # 70% 이상 유사하면 중복으로 간주

    def _update_processing_stats(self, results: List[ProcessingResult], total_time: float):
        """처리 통계 업데이트"""
        with self.processing_lock:
            successful_count = sum(1 for r in results if r.success)

            self.processing_stats["total_chunks_processed"] += len(results)
            self.processing_stats["successful_chunks"] += successful_count
            self.processing_stats["failed_chunks"] += len(results) - successful_count
            self.processing_stats["total_processing_time"] += total_time

            # 평균 처리 시간 계산
            if self.processing_stats["total_chunks_processed"] > 0:
                self.processing_stats["average_chunk_time"] = (
                    self.processing_stats["total_processing_time"] / self.processing_stats["total_chunks_processed"]
                )

    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        with self.processing_lock:
            return self.processing_stats.copy()

    def reset_processing_stats(self):
        """처리 통계 초기화"""
        with self.processing_lock:
            self.processing_stats = {
                "total_chunks_processed": 0,
                "successful_chunks": 0,
                "failed_chunks": 0,
                "total_processing_time": 0.0,
                "average_chunk_time": 0.0,
            }
