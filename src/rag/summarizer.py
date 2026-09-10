"""
계층적 요약 시스템

긴 문서나 대량 컨텍스트에서 핵심 내용만 추출하여
LLM이 효율적으로 처리할 수 있도록 돕는 요약 엔진.
"""

from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import re
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk


# NLTK 데이터 초기화
def _initialize_nltk():
    """NLTK 데이터 초기화"""
    try:
        # 필요한 데이터가 있는지 확인
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        return True
    except LookupError:
        try:
            # 필요한 데이터 다운로드
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return True
        except Exception:
            return False


# NLTK 초기화 시도
_nltk_available = _initialize_nltk()

try:
    if _nltk_available:
        from nltk.tokenize import sent_tokenize, word_tokenize
    else:
        raise ImportError("NLTK initialization failed")
except ImportError:
    # NLTK 데이터가 없는 경우 간단한 대체 구현
    def sent_tokenize(text):
        return re.split(r"[.!?]+", text)

    def word_tokenize(text):
        return re.findall(r"\w+", text.lower())

    _nltk_available = False

logger = logging.getLogger(__name__)


def resolve_summary_length(max_output_tokens: int, default_length: int = 500, min_length: int = 150) -> int:
    """모델 출력 토큰 한도에 맞춰 요약 목표 길이(문자)를 정한다.

    한글은 토큰 1개당 약 1문자로 잡아 안전하게 계산한다(ponytail: 보수적 가정,
    토크나이저별 정밀 계산이 필요해지면 token_counter로 교체).
    출력 한도가 기본값보다 크면 기본값을 유지하고, 작은 모델은 한도 내로 줄인다.
    """
    if max_output_tokens <= 0:
        return default_length
    return max(min_length, min(default_length, max_output_tokens))


class DocumentSummary:
    """문서 요약 결과를 담는 클래스"""

    def __init__(
        self,
        original_content: str,
        summary: str,
        key_sentences: List[str],
        importance_score: float,
        compression_ratio: float,
    ):
        self.original_content = original_content
        self.summary = summary
        self.key_sentences = key_sentences
        self.importance_score = importance_score
        self.compression_ratio = compression_ratio
        self.metadata = {}

    def get_compressed_content(self, target_length: int = None) -> str:
        """압축된 내용 반환"""
        if target_length is None:
            return self.summary

        # 목표 길이에 맞춰 조정
        if len(self.summary) <= target_length:
            return self.summary

        # 핵심 문장들을 우선순위에 따라 선택
        selected_sentences = []
        current_length = 0

        for sentence in self.key_sentences:
            if current_length + len(sentence) <= target_length:
                selected_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break

        return " ".join(selected_sentences) if selected_sentences else self.summary[:target_length]


class HierarchicalSummarizer:
    """계층적 요약 시스템"""

    def __init__(self, llm=None, max_summary_length: int = 500):
        """
        Args:
            llm: 요약에 사용할 LLM 모델
            max_summary_length: 기본 요약 최대 길이
        """
        self.llm = llm
        self.max_summary_length = max_summary_length
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))

        # 요약 프롬프트 템플릿
        self.summary_template = PromptTemplate(
            input_variables=["content", "query", "max_length"],
            template="""다음 문서의 핵심 내용을 {max_length}자 이내로 요약해주세요.

사용자 질문: {query}

문서 내용:
{content}

요약 시 다음 사항을 고려해주세요:
1. 사용자 질문과 관련된 내용을 우선적으로 포함
2. 중요한 사실, 숫자, 날짜 등은 정확히 유지
3. 핵심 개념과 주요 결론을 명확히 제시
4. 불필요한 반복이나 부가 설명은 제거

요약:""",
        )

    def create_document_summary(
        self, document: Document, query: str = "", target_length: int = None
    ) -> DocumentSummary:
        """단일 문서 요약 생성"""
        content = document.page_content
        target_length = target_length or self.max_summary_length

        # 1. 핵심 문장 추출
        key_sentences = self._extract_key_sentences(content, query, num_sentences=5)

        # 2. LLM 기반 요약 (가능한 경우)
        if self.llm:
            try:
                summary = self._generate_llm_summary(content, query, target_length)
            except Exception as e:
                logger.warning(f"LLM 요약 실패, 추출 요약 사용: {e}")
                summary = self._create_extractive_summary(key_sentences, target_length)
        else:
            summary = self._create_extractive_summary(key_sentences, target_length)

        # 3. 중요도 점수 계산
        importance_score = self._calculate_importance_score(content, query)

        # 4. 압축률 계산
        compression_ratio = len(summary) / len(content) if content else 0

        return DocumentSummary(
            original_content=content,
            summary=summary,
            key_sentences=key_sentences,
            importance_score=importance_score,
            compression_ratio=compression_ratio,
        )

    def create_multi_level_summary(
        self, documents: List[Tuple[Document, float]], query: str = "", levels: int = 3
    ) -> Dict[str, Any]:
        """다단계 계층적 요약 생성"""
        if not documents:
            return {"levels": [], "final_summary": "", "metadata": {}}

        logger.info(f"다단계 요약 시작: {len(documents)}개 문서, {levels}단계")

        summaries_by_level = {}
        current_docs = documents

        # 각 레벨별 요약 생성
        for level in range(levels):
            level_summaries = []

            # 현재 레벨의 문서들을 요약
            for doc, score in current_docs:
                doc_summary = self.create_document_summary(doc, query)
                doc_summary.metadata["level"] = level
                doc_summary.metadata["original_score"] = score
                level_summaries.append(doc_summary)

            summaries_by_level[f"level_{level}"] = level_summaries

            # 다음 레벨을 위한 준비 (요약본을 새로운 문서로 변환)
            if level < levels - 1:
                next_docs = []
                for summary in level_summaries:
                    new_doc = Document(page_content=summary.summary, metadata={"level": level + 1, "compressed": True})
                    next_docs.append((new_doc, summary.importance_score))
                current_docs = next_docs

        # 최종 요약 생성
        final_level_summaries = summaries_by_level[f"level_{levels-1}"]
        final_content = "\n\n".join([s.summary for s in final_level_summaries])

        if self.llm and len(final_content) > self.max_summary_length:
            final_summary = self._generate_llm_summary(final_content, query, self.max_summary_length)
        else:
            final_summary = final_content

        # 메타데이터 수집
        metadata = {
            "total_documents": len(documents),
            "levels_created": levels,
            "total_compression_ratio": len(final_summary) / sum(len(doc.page_content) for doc, _ in documents),
            "processing_statistics": self._calculate_processing_stats(summaries_by_level),
        }

        logger.info(f"다단계 요약 완료: 압축률 {metadata['total_compression_ratio']:.2%}")

        return {"levels": summaries_by_level, "final_summary": final_summary, "metadata": metadata}

    def compress_context(self, documents: List[Tuple[Document, float]], target_length: int, query: str = "") -> str:
        """컨텍스트를 목표 길이에 맞춰 압축"""
        if not documents:
            return ""

        total_original_length = sum(len(doc.page_content) for doc, _ in documents)

        if total_original_length <= target_length:
            # 압축이 필요 없는 경우
            return "\n\n".join([doc.page_content for doc, _ in documents])

        logger.info(f"컨텍스트 압축: {total_original_length:,}자 → {target_length:,}자")

        # 각 문서별 할당 길이 계산 (중요도 기반)
        doc_scores = [score for _, score in documents]
        total_score = sum(doc_scores)

        compressed_parts = []
        used_length = 0

        for i, (doc, score) in enumerate(documents):
            # 중요도에 비례하여 길이 할당
            if total_score > 0:
                allocated_length = int((score / total_score) * target_length)
            else:
                allocated_length = target_length // len(documents)

            # 최소 길이 보장
            allocated_length = max(100, allocated_length)

            # 남은 길이 확인
            remaining_length = target_length - used_length
            if remaining_length < allocated_length:
                allocated_length = remaining_length

            if allocated_length <= 0:
                break

            # 문서 요약
            doc_summary = self.create_document_summary(doc, query, allocated_length)
            compressed_content = doc_summary.get_compressed_content(allocated_length)

            compressed_parts.append(f"[문서 {i+1}]\n{compressed_content}")
            used_length += len(compressed_content)

            if used_length >= target_length:
                break

        result = "\n\n".join(compressed_parts)
        logger.info(f"압축 완료: {len(result):,}자 (목표: {target_length:,}자)")

        return result

    def _extract_key_sentences(self, content: str, query: str = "", num_sentences: int = 5) -> List[str]:
        """핵심 문장 추출"""
        if not content.strip():
            return []

        # 문장 분할
        sentences = sent_tokenize(content)
        if len(sentences) <= num_sentences:
            return sentences

        # TF-IDF 기반 문장 점수 계산
        vectorizer = TfidfVectorizer(stop_words="english")

        try:
            # 문장들을 벡터화
            sentence_vectors = vectorizer.fit_transform(sentences)

            # 쿼리가 있는 경우 쿼리와의 유사도 계산
            if query.strip():
                query_vector = vectorizer.transform([query])
                query_similarities = cosine_similarity(sentence_vectors, query_vector).flatten()
            else:
                query_similarities = np.zeros(len(sentences))

            # 문장별 TF-IDF 평균 점수 계산
            tfidf_scores = np.array(sentence_vectors.sum(axis=1)).flatten()

            # 최종 점수 = TF-IDF 점수 + 쿼리 유사도
            final_scores = tfidf_scores * 0.7 + query_similarities * 0.3

            # 상위 문장 선택
            top_indices = np.argsort(final_scores)[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # 원래 순서 유지

            return [sentences[i] for i in top_indices]

        except Exception as e:
            logger.warning(f"TF-IDF 기반 문장 추출 실패: {e}")
            # 간단한 길이 기반 선택으로 대체
            sentence_lengths = [(i, len(s)) for i, s in enumerate(sentences)]
            sentence_lengths.sort(key=lambda x: x[1], reverse=True)
            top_indices = sorted([i for i, _ in sentence_lengths[:num_sentences]])
            return [sentences[i] for i in top_indices]

    def _generate_llm_summary(self, content: str, query: str, max_length: int) -> str:
        """LLM을 사용한 요약 생성"""
        if not self.llm:
            raise ValueError("LLM이 설정되지 않았습니다.")

        # 프롬프트 생성
        prompt = self.summary_template.format(
            content=content[:10000],  # 입력 길이 제한
            query=query or "주요 내용 요약",
            max_length=max_length,
        )

        # LLM 호출
        response = self.llm.invoke(prompt)

        if hasattr(response, "content"):
            summary = response.content
        else:
            summary = str(response)

        # 길이 제한
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0] + "..."

        return summary.strip()

    def _create_extractive_summary(self, key_sentences: List[str], target_length: int) -> str:
        """추출 요약 생성"""
        if not key_sentences:
            return ""

        summary_parts = []
        current_length = 0

        for sentence in key_sentences:
            if current_length + len(sentence) <= target_length:
                summary_parts.append(sentence)
                current_length += len(sentence)
            else:
                # 마지막 문장을 부분적으로 포함
                remaining_length = target_length - current_length
                if remaining_length > 50:  # 최소 길이가 있는 경우만
                    partial_sentence = sentence[:remaining_length].rsplit(" ", 1)[0] + "..."
                    summary_parts.append(partial_sentence)
                break

        return " ".join(summary_parts)

    def _calculate_importance_score(self, content: str, query: str = "") -> float:
        """문서의 중요도 점수 계산"""
        if not content.strip():
            return 0.0

        # 기본 점수 (길이 기반)
        length_score = min(1.0, len(content) / 1000)

        # 쿼리 관련성 점수
        if query.strip():
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())

            if query_words:
                overlap_score = len(query_words.intersection(content_words)) / len(query_words)
            else:
                overlap_score = 0.0
        else:
            overlap_score = 0.0

        # 정보 밀도 점수 (숫자, 날짜, 고유명사 등)
        info_patterns = [
            r"\d{4}년",  # 연도
            r"\d+월",  # 월
            r"\d+일",  # 일
            r"\d+%",  # 퍼센트
            r"\d+개",  # 개수
            r"\d+명",  # 명수
        ]

        info_count = 0
        for pattern in info_patterns:
            info_count += len(re.findall(pattern, content))

        info_density_score = min(1.0, info_count / 10)

        # 최종 점수 (가중 평균)
        final_score = length_score * 0.3 + overlap_score * 0.5 + info_density_score * 0.2

        return final_score

    def _calculate_processing_stats(self, summaries_by_level: Dict[str, List[DocumentSummary]]) -> Dict[str, Any]:
        """처리 통계 계산"""
        stats = {}

        for level, summaries in summaries_by_level.items():
            level_stats = {
                "document_count": len(summaries),
                "avg_compression_ratio": np.mean([s.compression_ratio for s in summaries]),
                "avg_importance_score": np.mean([s.importance_score for s in summaries]),
                "total_original_length": sum(len(s.original_content) for s in summaries),
                "total_summary_length": sum(len(s.summary) for s in summaries),
            }
            stats[level] = level_stats

        return stats
