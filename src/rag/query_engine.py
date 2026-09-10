"""
쿼리 엔진 모듈

이 모듈은 RAG 시스템에서 쿼리 처리를 담당합니다.
쿼리 전처리, 검색, 컨텍스트 처리, 응답 생성을 포함합니다.
"""

from typing import List, Dict, Any, Generator
import logging

from config import settings
from src.utils import TextProcessor
from src.utils.answer_formatter import AnswerFormatter

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    쿼리 처리를 담당하는 엔진

    주요 기능:
    - 쿼리 전처리 및 확장
    - 벡터 검색 및 대안 검색
    - 표준/대용량 컨텍스트 처리
    - 스트리밍 응답 생성
    """

    def __init__(self, vector_db, llm_manager, document_processor):
        """
        쿼리 엔진 초기화

        Args:
            vector_db: 벡터 데이터베이스 인스턴스
            llm_manager: LLM 관리자 인스턴스
            document_processor: 문서 처리기 인스턴스
        """
        self.vector_db = vector_db
        self.llm_manager = llm_manager
        self.document_processor = document_processor
        self.answer_formatter = AnswerFormatter()
        self._last_context_tokens = 0

        # 체인 초기화는 외부에서 설정
        self.chain = None
        self.streaming_chain = None

    def set_chains(self, chain, streaming_chain):
        """LLM 체인 설정"""
        self.chain = chain
        self.streaming_chain = streaming_chain

    def query(self, question: str) -> Dict[str, Any]:
        """
        향상된 질문 처리 (대량 문서 지원)
        - 쿼리 전처리 및 확장
        - 검색 결과 분석
        - 컨텍스트 분할 및 병렬 처리 (필요시)
        - 컨텍스트 최적화
        """
        try:
            # 0. 쿼리 전처리 및 확장
            processed_question = self.preprocess_query(question)

            # 벡터 DB 상태 확인
            doc_count = self.vector_db.get_document_count()
            logger.info(f"벡터 DB 문서 수: {doc_count}")

            if doc_count == 0:
                return {
                    "answer": "벡터 데이터베이스에 문서가 없습니다. 먼저 문서를 업로드해주세요.",
                    "sources": [],
                    "status": "no_documents",
                    "search_info": self.get_search_info(question, []),
                }

            # 1. 관련 문서 검색 (향상된 검색 사용)
            k_docs = settings.k_documents

            relevant_docs = self.vector_db.search(processed_question, k=k_docs)
            logger.info(f"검색 결과: {len(relevant_docs)}개 문서 (처리 쿼리: '{processed_question[:50]}...')")

            if not relevant_docs:
                # 대안 검색 시도
                fallback_results = self.fallback_search(processed_question)

                if not fallback_results:
                    return {
                        "answer": self.generate_no_results_message(question, doc_count),
                        "sources": [],
                        "status": "no_relevant_documents",
                        "search_info": self.get_search_info(question, []),
                    }
                else:
                    relevant_docs = fallback_results

            # 2. 컨텍스트 처리 방식 결정
            total_content_length = sum(len(doc.page_content) for doc, _ in relevant_docs)

            # 대량 컨텍스트 처리 여부 결정
            enable_large_context = getattr(self, "enable_large_context_processing", True)
            large_context_threshold = getattr(self, "large_context_threshold", 50000)

            if enable_large_context and total_content_length > large_context_threshold:
                logger.info(f"대량 컨텍스트 감지 ({total_content_length:,}자) - 병렬 처리 모드")
                return self.process_large_context(question, relevant_docs)
            else:
                logger.info(f"표준 컨텍스트 처리 ({total_content_length:,}자)")
                return self.process_standard_context(question, relevant_docs)

        except Exception as e:
            logger.error(f"쿼리 처리 중 오류 발생 ({type(e).__name__}): {e}", exc_info=True)
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "sources": [],
                "status": "error",
                "search_info": {},
            }

    def stream_query(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """
        스트리밍 방식으로 질문 처리
        실시간으로 답변을 생성하여 yield 합니다.
        """
        try:
            # 0. 쿼리 전처리 및 확장
            processed_question = self.preprocess_query(question)

            # 벡터 DB 상태 확인
            doc_count = self.vector_db.get_document_count()
            logger.info(f"벡터 DB 문서 수: {doc_count}")

            if doc_count == 0:
                yield {
                    "type": "error",
                    "content": "벡터 데이터베이스에 문서가 없습니다. 먼저 문서를 업로드해주세요.",
                    "sources": [],
                    "status": "no_documents",
                }
                return

            # 검색 시작 알림
            yield {"type": "status", "content": "🔍 관련 문서를 검색하고 있습니다...", "status": "searching"}

            # 1. 관련 문서 검색
            k_docs = settings.k_documents

            relevant_docs = self.vector_db.search(processed_question, k=k_docs)
            logger.info(f"검색 결과: {len(relevant_docs)}개 문서")

            if not relevant_docs:
                # 대안 검색 시도
                fallback_results = self.fallback_search(processed_question)

                if not fallback_results:
                    yield {
                        "type": "error",
                        "content": self.generate_no_results_message(question, doc_count),
                        "sources": [],
                        "status": "no_relevant_documents",
                    }
                    return
                else:
                    relevant_docs = fallback_results

            # 검색 완료 및 답변 생성 시작 알림
            yield {
                "type": "status",
                "content": f"✅ {len(relevant_docs)}개 문서 발견. 답변을 생성하고 있습니다...",
                "status": "generating",
            }

            # 2. 컨텍스트 생성 (출처와 같은 문서 순서를 공유하도록 먼저 정렬)
            prepared_docs = self.document_processor.prepare_documents(relevant_docs)
            context = self.document_processor.format_documents(prepared_docs, question)
            self._last_context_tokens = len(context) // 4

            # 3. 스트리밍 방식 답변 생성
            full_response = ""

            # 스트리밍 체인 실행
            for chunk in self.streaming_chain.stream({"context": context, "question": question}):
                chunk_text = self._extract_chunk_text(chunk)
                if not chunk_text:
                    continue
                cleaned = self.document_processor.sanitize_output_chunk(chunk_text)
                full_response += cleaned
                formatted_preview = self.answer_formatter.format(full_response)
                yield {"type": "content", "content": cleaned, "full_content": formatted_preview, "status": "streaming"}

            # 4. 스트리밍 완료 후 최종 정보 전송 (컨텍스트 [문서 n]과 출처 번호 일치)
            sources = self.document_processor.generate_enhanced_sources(prepared_docs)
            search_info = self.get_search_info(question, relevant_docs)
            formatted_full = self.answer_formatter.format(full_response, sources)

            yield {
                "type": "complete",
                "content": formatted_full,
                "full_content": formatted_full,
                "sources": sources,
                "status": "success",
                "search_info": search_info,
                "context_tokens": self._last_context_tokens,
                "context_documents": [doc for doc, _ in relevant_docs],
            }

        except Exception as e:
            logger.error(
                f"스트리밍 쿼리 처리 중 오류 발생 ({type(e).__name__}): {e}",
                exc_info=True,
            )
            yield {
                "type": "error",
                "content": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "sources": [],
                "status": "error",
            }

    def preprocess_query(self, query: str) -> str:
        """향상된 쿼리 전처리 및 확장"""
        if not settings.enable_query_preprocessing:
            return query

        # 기본 정제
        processed_query = query.strip()

        # 특정 고유명사가 포함된 경우 제한적 확장
        specific_terms = ["우각형파수편", "고배", "토기편", "파수편"]
        contains_specific = any(term in processed_query for term in specific_terms)

        if settings.enable_query_expansion:
            if contains_specific:
                # 특정 용어가 있으면 기본 확장만 수행
                logger.info(f"특정 용어 감지, 제한적 확장 수행: {processed_query}")
                # 질문 의도 키워드만 추가
                if "뭐야" in processed_query or "무엇" in processed_query:
                    processed_query += " 백제 토기 유물"
                return processed_query
            else:
                # 1차: 새로운 동적 키워드 확장 사용 (임베딩 모델 포함)
                try:
                    # 임베딩 모델을 KeywordExpander에 전달
                    TextProcessor.get_keyword_expander(embedding_model=self.vector_db.embedding_model)

                    expanded_query = TextProcessor.expand_query(
                        processed_query,
                        use_static_expansion=False,  # 정적 확장은 비활성화
                        use_dynamic_expansion=True,  # 동적 확장 활성화
                        max_terms=20,  # 최대 20개 키워드
                    )

                    if expanded_query and expanded_query != processed_query:
                        logger.info(f"동적 키워드 확장 완료: '{query}' -> '{expanded_query[:100]}...'")
                        return expanded_query

                except Exception as e:
                    logger.warning(f"동적 키워드 확장 실패 ({type(e).__name__}): {e}")

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

                    processed_query = " ".join(unique_terms[:25])  # 최대 25개 단어
                    logger.info(f"문서 기반 확장 완료: '{query}' -> '{processed_query[:100]}...'")
                    return processed_query

            except Exception as e:
                logger.warning(f"문서 기반 확장 실패 ({type(e).__name__}): {e}")

            # 3차: 기존 정적 확장 사용 (폴백)
            try:
                processed_query = TextProcessor.expand_query(
                    processed_query, use_static_expansion=True, use_dynamic_expansion=False
                )
                logger.info(f"정적 확장 사용: '{query}' -> '{processed_query[:100]}...'")

            except Exception as e:
                logger.warning(f"정적 확장 실패 ({type(e).__name__}): {e}")

        logger.info(f"쿼리 전처리 완료: '{query}' -> '{processed_query[:100]}...'")
        return processed_query

    def fallback_search(self, query: str) -> List[tuple]:
        """검색 실패 시 대안 검색"""
        try:
            # 더 간단한 키워드로 검색
            simple_keywords = self.extract_keywords(query)
            if simple_keywords:
                fallback_query = " ".join(simple_keywords)
                logger.info(f"대안 검색 시도: '{fallback_query}'")
                return self.vector_db.search(fallback_query, k=5)
        except Exception as e:
            logger.error(f"대안 검색 실패 ({type(e).__name__}): {e}", exc_info=True)

        return []

    def extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        # 간단한 키워드 추출 (불용어 제거)
        stop_words = {
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "에서",
            "로",
            "으로",
            "와",
            "과",
            "의",
            "에게",
            "한테",
            "께",
            "뭐",
            "무엇",
            "어떤",
            "어떻게",
            "왜",
            "언제",
            "어디",
            "누구",
            "얼마나",
            "몇",
            "어느",
            "알려줘",
            "알려주세요",
            "설명해줘",
            "설명해주세요",
            "말해줘",
            "말해주세요",
            "가르쳐줘",
            "가르쳐주세요",
        }

        words = query.split()
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        return keywords[:5]  # 상위 5개만 반환

    def generate_no_results_message(self, question: str, doc_count: int) -> str:
        """검색 결과가 없을 때 메시지 생성"""
        return f"""죄송합니다. '{question}'에 대한 관련 문서를 찾을 수 없습니다.

현재 벡터 데이터베이스에는 {doc_count:,}개의 문서가 저장되어 있습니다.

다음 사항을 확인해보세요:
1. 질문을 다른 방식으로 표현해보세요
2. 더 일반적인 키워드를 사용해보세요
3. 맞춤법이나 띄어쓰기를 확인해보세요

예시:
- '몽촌토성에 대해 알려줘' → '몽촌토성 백제'
- '발굴조사 결과는?' → '발굴 유물 출토'"""

    def process_standard_context(self, question: str, relevant_docs: List[tuple]) -> Dict[str, Any]:
        """기존 방식의 표준 컨텍스트 처리"""
        try:
            # 컨텍스트와 출처가 같은 문서 순서를 공유하도록 먼저 정렬한다
            prepared_docs = self.document_processor.prepare_documents(relevant_docs)
            context = self.document_processor.format_documents(prepared_docs, question)
            self._last_context_tokens = len(context) // 4

            # LLM 체인 실행
            response = self.chain.invoke({"context": context, "question": question})

            # 응답 텍스트 추출
            if hasattr(response, "content"):
                answer = response.content
            elif isinstance(response, dict) and "text" in response:
                answer = response["text"]
            elif isinstance(response, str):
                answer = response
            else:
                answer = str(response)

            # 출처 정보 생성 ([문서 n] 라벨과 [출처 n] 번호를 일치시키기 위해 정렬된 문서 사용)
            sources = self.document_processor.generate_enhanced_sources(prepared_docs)
            search_info = self.get_search_info(question, relevant_docs)
            formatted_answer = self.answer_formatter.format(answer, sources)

            context_documents = [doc for doc, _ in relevant_docs]
            return {
                "answer": formatted_answer,
                "sources": sources,
                "status": "success",
                "search_info": search_info,
                "context_tokens": self._last_context_tokens,
                "context_documents": context_documents,
            }

        except Exception as e:
            logger.error(f"표준 컨텍스트 처리 중 오류: {str(e)}")
            raise

    def _extract_chunk_text(self, chunk: Any) -> str:
        """스트리밍 청크에서 텍스트를 추출한다."""
        if hasattr(chunk, "content"):
            return str(chunk.content)
        if isinstance(chunk, dict):
            if "text" in chunk:
                return str(chunk["text"])
            if "content" in chunk:
                return str(chunk["content"])
        if isinstance(chunk, str):
            return chunk
        return str(chunk)

    def process_large_context(self, question: str, relevant_docs: List[tuple]) -> Dict[str, Any]:
        """대량 컨텍스트 처리 (병렬 처리 사용)"""
        try:
            # 대량 컨텍스트 처리 로직은 복잡하므로 여기서는 표준 처리로 폴백
            logger.warning("대량 컨텍스트 처리는 아직 구현되지 않음. 표준 처리로 폴백합니다.")
            return self.process_standard_context(question, relevant_docs)

        except Exception as e:
            logger.error(f"대량 컨텍스트 처리 중 오류: {str(e)}")
            raise

    def get_search_info(self, question: str, documents: List[tuple]) -> Dict[str, Any]:
        """검색 정보 생성"""
        return {
            "original_query": question,
            "processed_query": self.preprocess_query(question),
            "total_documents": len(documents),
            "search_method": settings.vector_db_type,
            "context_tokens": self._last_context_tokens,
        }

    def get_last_context_tokens(self) -> int:
        """마지막 쿼리에서 사용된 컨텍스트 토큰 수 반환"""
        return self._last_context_tokens

    def search_documents(self, query: str, k: int = 8) -> List[tuple]:
        """
        벡터 DB에서 문서 검색

        Args:
            query: 검색 쿼리
            k: 검색할 문서 수

        Returns:
            (Document, score) 튜플 리스트
        """
        try:
            # 쿼리 전처리
            processed_query = self.preprocess_query(query)

            # 벡터 DB에서 검색
            documents = self.vector_db.search(processed_query, k)

            logger.info(f"문서 검색 완료: 쿼리='{query}', 결과={len(documents)}개")
            return documents

        except Exception as e:
            logger.error(f"문서 검색 중 오류 ({type(e).__name__}): {e}", exc_info=True)
            return []
