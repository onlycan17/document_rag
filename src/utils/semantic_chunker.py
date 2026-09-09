#!/usr/bin/env python3
"""
의미 기반 청킹 알고리즘 모듈

주요 기능:
- 키워드 기반 주제 전환점 감지
- 문단 간 유사도 측정을 통한 의미적 경계 식별
- 적응적 청크 크기 조정
- 한글 텍스트 특성을 고려한 처리
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

logger = logging.getLogger(__name__)

# 주제 전환 키워드 정의 (한국어 고고학/역사 문서 특화)
TOPIC_TRANSITION_KEYWORDS = {
    "시간적_전환": [
        "그러나",
        "한편",
        "이후",
        "다음으로",
        "그 후",
        "뒤이어",
        "연이어",
        "시기",
        "시대",
        "년도",
        "세기",
        "기원전",
        "기원후",
        "당시",
        "이전",
        "동시에",
        "같은 시기",
        "그 시절",
    ],
    "공간적_전환": [
        "반면",
        "다른 지역",
        "인근",
        "주변",
        "동쪽",
        "서쪽",
        "남쪽",
        "북쪽",
        "지역",
        "위치",
        "장소",
        "구역",
        "지점",
        "현장",
        "터",
        "지구",
    ],
    "논리적_전환": [
        "따라서",
        "그러므로",
        "결과적으로",
        "결론적으로",
        "요약하면",
        "즉",
        "다시 말해",
        "예를 들어",
        "구체적으로",
        "특히",
        "특별히",
    ],
    "대조적_전환": [
        "그러나",
        "하지만",
        "반면에",
        "이와 달리",
        "이에 반해",
        "대조적으로",
        "반대로",
        "한편으로는",
        "다른 한편으로는",
    ],
    "구조적_전환": [
        "첫째",
        "둘째",
        "셋째",
        "마지막으로",
        "다음",
        "또한",
        "그리고",
        "제1장",
        "제2장",
        "제1절",
        "제2절",
        "가.",
        "나.",
        "1)",
        "2)",
    ],
}

# 고고학/역사 전문 용어 (가중치 부여)
DOMAIN_KEYWORDS = {
    "시대_분류": ["구석기", "신석기", "청동기", "철기", "삼국시대", "통일신라", "고려", "조선"],
    "유적_유물": ["토기", "석기", "청동기", "철기", "유적", "유물", "유구", "출토품"],
    "발굴_조사": ["발굴", "조사", "탐사", "시굴", "확인조사", "구제발굴", "시굴조사"],
    "분석_방법": ["방사성탄소", "연대측정", "형식학", "층위", "편년", "분류", "분석"],
    "지명_지역": ["몽촌토성", "풍납토성", "한성", "백제", "고구려", "신라"],
}


class SemanticChunker:
    """의미 기반 청크 분할 클래스"""

    def __init__(self, min_chunk_size: int = 300, max_chunk_size: int = 1500, similarity_threshold: float = 0.3):
        """
        SemanticChunker 초기화

        Args:
            min_chunk_size: 최소 청크 크기 (문자 수)
            max_chunk_size: 최대 청크 크기 (문자 수)
            similarity_threshold: 문단 간 유사도 임계값 (0.0 ~ 1.0)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.topic_transition_keywords = {key: list(words) for key, words in TOPIC_TRANSITION_KEYWORDS.items()}
        self.domain_keywords = {key: list(words) for key, words in DOMAIN_KEYWORDS.items()}

    def create_semantic_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        텍스트를 의미 기반으로 청크 분할

        Args:
            text: 분할할 텍스트
            metadata: 추가 메타데이터

        Returns:
            List[Dict]: 청크 정보 리스트 [{'text': str, 'metadata': dict, 'semantic_info': dict}]
        """
        if not text.strip():
            return []

        logger.info(f"의미 기반 청킹 시작: {len(text)} 문자")

        # 1단계: 텍스트를 문단으로 분할
        paragraphs = self._split_into_paragraphs(text)
        logger.debug(f"문단 분할 완료: {len(paragraphs)}개 문단")

        # 2단계: 각 문단의 주제 키워드 추출
        paragraph_keywords = []
        for i, paragraph in enumerate(paragraphs):
            keywords = self._extract_keywords(paragraph)
            paragraph_keywords.append(keywords)
            logger.debug(f"문단 {i+1} 키워드: {keywords[:5]}...")  # 상위 5개만 로그

        # 3단계: 주제 전환점 감지
        transition_points = self._detect_topic_transitions(paragraphs, paragraph_keywords)
        logger.debug(f"주제 전환점 감지: {transition_points}")

        # 4단계: 문단 간 유사도 기반 경계 조정
        adjusted_boundaries = self._adjust_boundaries_by_similarity(paragraphs, paragraph_keywords, transition_points)
        logger.debug(f"유사도 기반 경계 조정: {adjusted_boundaries}")

        # 5단계: 적응적 청크 크기 조정으로 최종 청크 생성
        chunks = self._create_adaptive_chunks(paragraphs, adjusted_boundaries, paragraph_keywords)

        # 6단계: 청크 메타데이터 생성
        chunk_objects = []
        for i, (chunk_text, chunk_keywords) in enumerate(chunks):
            semantic_info = {
                "chunk_id": i,
                "keywords": chunk_keywords,
                "paragraph_count": len([p for p in paragraphs if p in chunk_text]),
                "topic_coherence": self._calculate_topic_coherence(chunk_keywords),
                "domain_relevance": self._calculate_domain_relevance(chunk_keywords),
            }

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update(
                {
                    "chunk_method": "semantic",
                    "chunk_size": len(chunk_text),
                    "semantic_score": semantic_info["topic_coherence"],
                }
            )

            chunk_objects.append({"text": chunk_text, "metadata": chunk_metadata, "semantic_info": semantic_info})

        logger.info(f"의미 기반 청킹 완료: {len(chunk_objects)}개 청크 생성")
        return chunk_objects

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 의미적 문단으로 분할"""
        # 기본적인 문단 분할 (빈 줄 기준)
        basic_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # 문단이 너무 긴 경우 추가 분할
        refined_paragraphs = []
        for paragraph in basic_paragraphs:
            if len(paragraph) > self.max_chunk_size:
                # 문장 단위로 분할
                sentences = self._split_into_sentences(paragraph)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + sentence) > self.max_chunk_size and current_chunk:
                        refined_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += (" " if current_chunk else "") + sentence

                if current_chunk:
                    refined_paragraphs.append(current_chunk.strip())
            else:
                refined_paragraphs.append(paragraph)

        return refined_paragraphs

    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분할 (한국어 특화)"""
        # 한국어 문장 종결 패턴
        sentence_endings = r"[.!?]|다\.|음\.|였다\.|었다\.|한다\.|된다\.|이다\.|않다\.|있다\.|없다\."

        # 문장 분할
        sentences = re.split(f"({sentence_endings})", text)

        # 분할된 부분을 다시 조합
        result = []
        current_sentence = ""

        for i, part in enumerate(sentences):
            current_sentence += part

            # 문장 종결 패턴에 매치되거나 마지막 부분인 경우
            if re.match(sentence_endings, part) or (i == len(sentences) - 1 and current_sentence.strip()):
                if current_sentence.strip():
                    result.append(current_sentence.strip())
                current_sentence = ""

        return result

    def _extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """텍스트에서 키워드 추출 및 가중치 계산"""
        # 1단계: 기본 단어 추출
        words = re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}|\d+", text)
        word_freq = Counter(words)

        # 2단계: TF 계산
        total_words = len(words)
        tf_scores = {word: count / total_words for word, count in word_freq.items()}

        # 3단계: 도메인 특화 가중치 적용
        weighted_scores = {}
        for word, tf_score in tf_scores.items():
            weight = 1.0

            # 도메인 키워드 가중치
            for category, keywords in self.domain_keywords.items():
                if word in keywords:
                    weight *= 2.0
                    break

            # 길이 기반 가중치 (너무 짧거나 긴 단어 패널티)
            if len(word) < 2:
                weight *= 0.5
            elif len(word) > 10:
                weight *= 0.8

            # 숫자만으로 구성된 경우 가중치 감소
            if word.isdigit():
                weight *= 0.7

            weighted_scores[word] = tf_score * weight

        # 4단계: 상위 키워드 반환 (가중치 순)
        sorted_keywords = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:20]  # 상위 20개만 반환

    def _detect_topic_transitions(
        self, paragraphs: List[str], paragraph_keywords: List[List[Tuple[str, float]]]
    ) -> List[int]:
        """주제 전환점 감지"""
        transition_points = []

        for i in range(1, len(paragraphs)):
            transition_score = 0.0
            paragraph = paragraphs[i]

            # 1. 키워드 기반 전환점 감지
            for category, keywords in self.topic_transition_keywords.items():
                for keyword in keywords:
                    if keyword in paragraph:
                        if category == "구조적_전환":
                            transition_score += 2.0
                        elif category == "논리적_전환":
                            transition_score += 1.5
                        else:
                            transition_score += 1.0
                        break

            # 2. 키워드 유사도 기반 전환점 감지
            current_keywords = set(word for word, _ in paragraph_keywords[i])
            prev_keywords = set(word for word, _ in paragraph_keywords[i - 1])

            if current_keywords and prev_keywords:
                similarity = len(current_keywords & prev_keywords) / len(current_keywords | prev_keywords)
                if similarity < self.similarity_threshold:
                    transition_score += (1.0 - similarity) * 2.0

            # 3. 구조적 패턴 기반 전환점 감지
            structure_patterns = [
                r"^제\s*\d+\s*[장절]",  # 제1장, 제2절
                r"^\d+\.\s+[가-힣]",  # 1. 서론
                r"^[가-힣]\.\s+[가-힣]",  # 가. 개요
                r"^【[^】]+】",  # 【제목】
                r"^##+\s",  # 마크다운 헤딩
            ]

            for pattern in structure_patterns:
                if re.match(pattern, paragraph):
                    transition_score += 3.0
                    break

            # 임계값을 넘으면 전환점으로 판단
            if transition_score >= 2.0:
                transition_points.append(i)

        return transition_points

    def _adjust_boundaries_by_similarity(
        self, paragraphs: List[str], paragraph_keywords: List[List[Tuple[str, float]]], initial_boundaries: List[int]
    ) -> List[int]:
        """문단 간 유사도를 고려하여 경계 조정"""
        adjusted_boundaries = initial_boundaries.copy()

        # 인접한 문단 간 유사도 계산 및 경계 조정
        for i in range(len(paragraphs) - 1):
            if i in adjusted_boundaries:
                continue

            current_keywords = set(word for word, _ in paragraph_keywords[i])
            next_keywords = set(word for word, _ in paragraph_keywords[i + 1])

            if current_keywords and next_keywords:
                similarity = len(current_keywords & next_keywords) / len(current_keywords | next_keywords)

                # 유사도가 높으면 경계 제거 (연결)
                if similarity > 0.7:
                    # 이미 경계가 설정된 경우 제거
                    if (i + 1) in adjusted_boundaries:
                        adjusted_boundaries.remove(i + 1)

                # 유사도가 낮으면 경계 추가 (분할)
                elif similarity < 0.2:
                    if (i + 1) not in adjusted_boundaries:
                        adjusted_boundaries.append(i + 1)

        return sorted(adjusted_boundaries)

    def _create_adaptive_chunks(
        self, paragraphs: List[str], boundaries: List[int], paragraph_keywords: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, List[Tuple[str, float]]]]:
        """적응적 청크 크기 조정을 통한 최종 청크 생성"""
        chunks = []
        start_idx = 0

        # 경계점에 마지막 인덱스 추가
        boundaries = boundaries + [len(paragraphs)]

        for boundary in boundaries:
            # 현재 청크 텍스트 구성
            chunk_paragraphs = paragraphs[start_idx:boundary]
            chunk_text = "\n\n".join(chunk_paragraphs)

            # 현재 청크의 키워드 통합
            chunk_keywords = {}
            for i in range(start_idx, boundary):
                for word, score in paragraph_keywords[i]:
                    chunk_keywords[word] = chunk_keywords.get(word, 0) + score

            # 키워드를 점수 순으로 정렬
            sorted_chunk_keywords = sorted(chunk_keywords.items(), key=lambda x: x[1], reverse=True)

            # 청크 크기 확인 및 조정
            if len(chunk_text) < self.min_chunk_size and chunks:
                # 이전 청크와 병합
                prev_text, prev_keywords = chunks[-1]
                merged_text = prev_text + "\n\n" + chunk_text

                # 키워드 병합
                merged_keywords = {}
                for word, score in prev_keywords + sorted_chunk_keywords:
                    merged_keywords[word] = merged_keywords.get(word, 0) + score

                final_merged_keywords = sorted(merged_keywords.items(), key=lambda x: x[1], reverse=True)
                chunks[-1] = (merged_text, final_merged_keywords)

            elif len(chunk_text) > self.max_chunk_size:
                # 청크가 너무 큰 경우 문장 단위로 재분할
                sentences = self._split_into_sentences(chunk_text)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + sentence) > self.max_chunk_size and current_chunk:
                        # 현재 청크의 키워드 추출
                        sent_keywords = self._extract_keywords(current_chunk)
                        chunks.append((current_chunk.strip(), sent_keywords))
                        current_chunk = sentence
                    else:
                        current_chunk += (" " if current_chunk else "") + sentence

                if current_chunk:
                    sent_keywords = self._extract_keywords(current_chunk)
                    chunks.append((current_chunk.strip(), sent_keywords))
            else:
                # 적절한 크기의 청크
                chunks.append((chunk_text, sorted_chunk_keywords))

            start_idx = boundary

        return chunks

    def _calculate_topic_coherence(self, keywords: List[Tuple[str, float]]) -> float:
        """주제 일관성 점수 계산"""
        if not keywords:
            return 0.0

        # 상위 키워드들의 점수 분포를 기반으로 일관성 계산
        scores = [score for _, score in keywords[:10]]  # 상위 10개 키워드

        if len(scores) < 2:
            return 0.5

        # 점수의 표준편차를 이용한 일관성 측정
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)

        # 표준편차가 작을수록 일관성이 높음
        coherence = 1.0 / (1.0 + std_dev)
        return min(1.0, coherence)

    def _calculate_domain_relevance(self, keywords: List[Tuple[str, float]]) -> float:
        """도메인 관련성 점수 계산"""
        if not keywords:
            return 0.0

        domain_score = 0.0
        len(keywords)

        for word, score in keywords:
            for category, domain_words in self.domain_keywords.items():
                if word in domain_words:
                    domain_score += score
                    break

        # 전체 키워드 점수 대비 도메인 관련 키워드 점수 비율
        total_score = sum(score for _, score in keywords)
        if total_score > 0:
            relevance = domain_score / total_score
        else:
            relevance = 0.0

        return min(1.0, relevance)

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """청크 통계 정보 반환"""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk["text"]) for chunk in chunks]
        coherence_scores = [chunk["semantic_info"]["topic_coherence"] for chunk in chunks]
        domain_scores = [chunk["semantic_info"]["domain_relevance"] for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_coherence": sum(coherence_scores) / len(coherence_scores),
            "avg_domain_relevance": sum(domain_scores) / len(domain_scores),
            "total_keywords": sum(len(chunk["semantic_info"]["keywords"]) for chunk in chunks),
        }
