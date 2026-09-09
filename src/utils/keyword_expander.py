"""
동적 키워드 확장 모듈

사용자 질문에서 핵심 단어를 추출하고 동의어, 유사어, 연관어를 추가하여
검색 성능을 향상시키는 기능을 제공합니다.
"""

from typing import List, Dict
import re
import logging

from .keyword_data import DOMAIN_SPECIFIC_TERMS, KOREAN_STOPWORDS, SYNONYM_DICT

logger = logging.getLogger(__name__)


class KeywordExpander:
    """
    동적 키워드 확장 클래스

    다양한 기법을 조합하여 사용자 쿼리의 키워드를 확장:
    1. 한국어 동의어/유사어 사전
    2. 의미 기반 유사어 추출
    3. 문맥 기반 연관어 추출
    4. 도메인별 전문 용어 매핑
    """

    def __init__(self, embedding_model=None):
        """
        키워드 확장기 초기화

        Args:
            embedding_model: 임베딩 모델 인스턴스 (옵션, 의미 기반 확장용)
        """
        self.synonym_dict = SYNONYM_DICT
        self.domain_terms = DOMAIN_SPECIFIC_TERMS
        self.stopwords = KOREAN_STOPWORDS
        self.embedding_model = embedding_model

        # 의미 기반 확장을 위한 캐시
        self.semantic_cache = {}

        logger.info("키워드 확장기 초기화 완료")

    def expand_keywords(
        self, query: str, max_terms: int = 25, use_semantic: bool = True, use_domain: bool = True
    ) -> List[str]:
        """
        쿼리의 키워드를 동적으로 확장

        Args:
            query: 확장할 쿼리
            max_terms: 최대 용어 개수
            use_semantic: 의미 기반 확장 사용 여부
            use_domain: 도메인별 확장 사용 여부

        Returns:
            확장된 키워드 리스트
        """
        try:
            if not query or not query.strip():
                return []

            # 1. 원본 쿼리에서 핵심 키워드 추출
            core_keywords = self._extract_core_keywords(query)
            logger.debug(f"핵심 키워드 추출: {core_keywords}")

            # 2. 동의어/유사어 확장
            synonym_keywords = self._expand_with_synonyms(core_keywords)
            logger.debug(f"동의어 확장: {synonym_keywords}")

            # 3. 의미 기반 확장 (선택적)
            semantic_keywords = []
            if use_semantic:
                semantic_keywords = self._expand_with_semantics(query, core_keywords)
                logger.debug(f"의미 기반 확장: {semantic_keywords}")

            # 4. 도메인별 확장 (선택적)
            domain_keywords = []
            if use_domain:
                domain_keywords = self._expand_with_domain_terms(core_keywords)
                logger.debug(f"도메인별 확장: {domain_keywords}")

            # 5. 모든 키워드 통합 및 정제
            all_keywords = self._combine_and_filter_keywords(
                core_keywords, synonym_keywords, semantic_keywords, domain_keywords
            )

            # 6. 중요도 기반 정렬 및 제한
            final_keywords = self._rank_and_limit_keywords(all_keywords, query, max_terms)

            logger.info(f"키워드 확장 완료: '{query}' -> {len(final_keywords)}개 키워드")
            return final_keywords

        except Exception as e:
            logger.error(f"키워드 확장 중 오류 발생: {str(e)}")
            # 오류 시 원본 쿼리의 단어들만 반환
            return [word for word in query.split() if len(word) > 1]

    def _extract_core_keywords(self, query: str) -> List[str]:
        """
        쿼리에서 핵심 키워드 추출

        Args:
            query: 분석할 쿼리

        Returns:
            핵심 키워드 리스트
        """
        # 텍스트 정제
        cleaned_query = re.sub(r"[^\w\s가-힣]", " ", query)
        words = cleaned_query.split()

        # 불용어 제거 및 길이 필터링
        core_keywords = []
        for word in words:
            word = word.strip()
            if len(word) > 1 and word not in self.stopwords and not word.isdigit():
                core_keywords.append(word)

        return core_keywords

    def _expand_with_synonyms(self, keywords: List[str]) -> List[str]:
        """
        동의어/유사어로 키워드 확장

        Args:
            keywords: 원본 키워드 리스트

        Returns:
            확장된 키워드 리스트
        """
        expanded = []

        for keyword in keywords:
            # 완전 일치
            if keyword in self.synonym_dict:
                expanded.extend(self.synonym_dict[keyword])

            # 부분 일치 (키워드가 사전의 키에 포함되거나 포함하는 경우)
            for dict_key, synonyms in self.synonym_dict.items():
                if (keyword in dict_key or dict_key in keyword) and keyword != dict_key:
                    expanded.extend(synonyms[:3])  # 부분 일치는 최대 3개만

        return expanded

    def _expand_with_semantics(self, query: str, keywords: List[str]) -> List[str]:
        """
        의미 기반 키워드 확장

        Args:
            query: 원본 쿼리
            keywords: 핵심 키워드들

        Returns:
            의미적으로 관련된 키워드 리스트
        """
        semantic_keywords = []

        # 1. 질문 의도 기반 확장
        intent = self._analyze_query_intent(query)
        intent_keywords = self._get_intent_keywords(intent)
        semantic_keywords.extend(intent_keywords)

        # 2. 임베딩 기반 의미적 유사어 추출 (가능한 경우)
        if self.embedding_model:
            embedding_keywords = self._extract_semantic_similar_words(keywords)
            semantic_keywords.extend(embedding_keywords)

        # 3. 키워드 조합으로 복합어 생성 (비활성화 - 의미 없는 조합 방지)
        # compound_keywords = self._generate_compound_keywords(keywords)
        # semantic_keywords.extend(compound_keywords)

        # 4. 문맥 기반 연관어 추출
        contextual_keywords = self._extract_contextual_keywords(query, keywords)
        semantic_keywords.extend(contextual_keywords)

        return semantic_keywords[:15]  # 최대 15개로 제한

    def _get_intent_keywords(self, intent: str) -> List[str]:
        """의도별 관련 키워드 반환"""
        intent_mappings = {
            "what": ["정의", "개념", "의미", "설명", "내용", "특징", "성질"],
            "how": ["방법", "절차", "과정", "단계", "수행", "방식", "기법"],
            "when": ["시기", "기간", "일정", "날짜", "시점", "언제", "시간"],
            "where": ["위치", "장소", "곳", "지역", "영역", "어디", "공간"],
            "why": ["이유", "원인", "목적", "배경", "근거", "까닭", "동기"],
            "who": ["담당자", "책임자", "관리자", "주체", "기관", "누구", "담당"],
        }
        return intent_mappings.get(intent, [])

    def _extract_semantic_similar_words(self, keywords: List[str]) -> List[str]:
        """
        임베딩 기반 의미적 유사어 추출

        Args:
            keywords: 대상 키워드들

        Returns:
            의미적으로 유사한 키워드들
        """
        if not self.embedding_model or not keywords:
            return []

        similar_words = []

        try:
            # 캐시 확인
            cache_key = tuple(sorted(keywords))
            if cache_key in self.semantic_cache:
                return self.semantic_cache[cache_key]

            # 동의어 사전에서 의미적으로 관련된 단어들 추출
            for keyword in keywords:
                # 해당 키워드와 관련된 모든 동의어 수집
                related_words = set()

                # 직접 매칭
                if keyword in self.synonym_dict:
                    related_words.update(self.synonym_dict[keyword][:5])

                # 키워드가 포함된 다른 용어들 찾기
                for dict_key, synonyms in self.synonym_dict.items():
                    if keyword in dict_key and keyword != dict_key:
                        related_words.update(synonyms[:3])
                    # 동의어 목록에 키워드가 포함된 경우
                    elif keyword in synonyms:
                        related_words.add(dict_key)
                        related_words.update(synonyms[:3])

                similar_words.extend(list(related_words))

            # 결과 캐싱
            result = list(set(similar_words))[:10]
            self.semantic_cache[cache_key] = result

            return result

        except Exception as e:
            logger.debug(f"의미적 유사어 추출 실패: {str(e)}")
            return []

    def _generate_compound_keywords(self, keywords: List[str]) -> List[str]:
        """키워드 조합으로 복합어 생성"""
        compound_keywords = []

        if len(keywords) >= 2:
            for i, keyword1 in enumerate(keywords):
                for keyword2 in keywords[i + 1 :]:
                    # 순서 고려한 조합
                    combined1 = f"{keyword1}{keyword2}"
                    combined2 = f"{keyword2}{keyword1}"

                    # 적절한 길이의 조합어만 추가
                    if 3 <= len(combined1) <= 12:
                        compound_keywords.append(combined1)
                    if 3 <= len(combined2) <= 12 and combined2 != combined1:
                        compound_keywords.append(combined2)

        return compound_keywords[:8]  # 최대 8개로 제한

    def _extract_contextual_keywords(self, query: str, keywords: List[str]) -> List[str]:
        """
        문맥 기반 연관어 추출

        Args:
            query: 원본 쿼리
            keywords: 핵심 키워드들

        Returns:
            문맥적으로 관련된 키워드들
        """
        contextual_keywords = []

        # 문서/정보 관련 쿼리인지 확인
        doc_related = any(word in query.lower() for word in ["문서", "자료", "정보", "데이터"])
        if doc_related:
            contextual_keywords.extend(["문서", "자료", "정보", "데이터", "파일"])

        # 시스템/기술 관련 쿼리인지 확인
        tech_related = any(word in query.lower() for word in ["시스템", "기술", "프로그램", "소프트웨어"])
        if tech_related:
            contextual_keywords.extend(["시스템", "기술", "솔루션", "프로그램", "개발"])

        # 관리/운영 관련 쿼리인지 확인
        mgmt_related = any(word in query.lower() for word in ["관리", "운영", "정책", "절차"])
        if mgmt_related:
            contextual_keywords.extend(["관리", "운영", "정책", "절차", "프로세스"])

        # 법령/규정 관련 쿼리인지 확인
        legal_related = any(word in query.lower() for word in ["법", "규정", "지침", "가이드"])
        if legal_related:
            contextual_keywords.extend(["법령", "규정", "지침", "가이드라인", "기준"])

        # 고고학/역사 관련 쿼리인지 확인
        archaeology_related = any(
            word in query for word in ["토기", "유물", "발굴", "유적", "파수", "고배", "옹", "호", "시루"]
        )
        if archaeology_related:
            contextual_keywords.extend(["유물", "토기", "발굴", "유적", "백제", "고구려", "출토품"])

        # 몽촌토성 관련 쿼리인지 확인
        mongchon_related = any(word in query for word in ["몽촌토성", "몽촌", "토성"])
        if mongchon_related:
            contextual_keywords.extend(["몽촌토성", "백제", "한성", "왕성", "토성", "발굴조사"])

        return list(set(contextual_keywords))[:12]  # 중복 제거 후 최대 12개

    def _expand_with_domain_terms(self, keywords: List[str]) -> List[str]:
        """
        도메인별 전문 용어로 확장

        Args:
            keywords: 핵심 키워드들

        Returns:
            도메인별 관련 용어 리스트
        """
        domain_keywords = []

        # 키워드별로 해당하는 도메인 찾기
        for keyword in keywords:
            for domain, terms in self.domain_terms.items():
                if keyword in terms:
                    # 같은 도메인의 다른 용어들 추가
                    domain_keywords.extend([term for term in terms if term != keyword])

        return list(set(domain_keywords))  # 중복 제거

    def _analyze_query_intent(self, query: str) -> str:
        """
        쿼리의 의도 분석

        Args:
            query: 분석할 쿼리

        Returns:
            의도 분류 (what, how, when, where, why, who, other)
        """
        query_lower = query.lower()

        # 의도별 키워드 패턴
        intent_patterns = {
            "what": ["무엇", "뭐", "어떤", "정의", "개념", "의미"],
            "how": ["어떻게", "방법", "어떤방법", "어떤식", "과정", "절차"],
            "when": ["언제", "시기", "기간", "일정", "시점"],
            "where": ["어디", "장소", "위치", "곳"],
            "why": ["왜", "이유", "원인", "목적", "까닭"],
            "who": ["누가", "누구", "담당", "책임"],
        }

        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent

        return "other"

    def _combine_and_filter_keywords(self, *keyword_lists: List[List[str]]) -> List[str]:
        """
        여러 키워드 리스트를 결합하고 필터링

        Args:
            keyword_lists: 키워드 리스트들

        Returns:
            결합되고 필터링된 키워드 리스트
        """
        all_keywords = []
        for keyword_list in keyword_lists:
            all_keywords.extend(keyword_list)

        # 중복 제거 (순서 유지)
        seen = set()
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in seen and len(keyword) > 1 and keyword not in self.stopwords:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def _rank_and_limit_keywords(self, keywords: List[str], original_query: str, max_terms: int) -> List[str]:
        """
        키워드를 중요도 순으로 정렬하고 개수 제한

        Args:
            keywords: 정렬할 키워드 리스트
            original_query: 원본 쿼리
            max_terms: 최대 키워드 개수

        Returns:
            정렬되고 제한된 키워드 리스트
        """
        # 키워드별 점수 계산
        keyword_scores = {}
        original_words = set(original_query.split())

        for keyword in keywords:
            score = 0

            # 1. 원본 쿼리에 포함되면 높은 점수
            if keyword in original_words:
                score += 10

            # 2. 길이가 적절하면 보너스 (2-8자)
            if 2 <= len(keyword) <= 8:
                score += 3
            elif len(keyword) > 8:
                score -= 2

            # 3. 한국어 비율이 높으면 보너스
            korean_ratio = sum(1 for c in keyword if "\uac00" <= c <= "\ud7a3") / len(keyword)
            score += korean_ratio * 2

            # 4. 숫자가 포함되면 약간 감점
            if any(c.isdigit() for c in keyword):
                score -= 1

            keyword_scores[keyword] = score

        # 점수 순으로 정렬
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 키워드들 반환
        result = [keyword for keyword, score in sorted_keywords[:max_terms]]

        return result

    def get_expansion_stats(self, query: str) -> Dict[str, any]:
        """
        키워드 확장 통계 정보 반환

        Args:
            query: 분석할 쿼리

        Returns:
            확장 통계 정보
        """
        core_keywords = self._extract_core_keywords(query)
        expanded_keywords = self.expand_keywords(query)

        return {
            "original_query": query,
            "core_keywords": core_keywords,
            "core_keyword_count": len(core_keywords),
            "expanded_keywords": expanded_keywords,
            "expanded_keyword_count": len(expanded_keywords),
            "expansion_ratio": len(expanded_keywords) / max(len(core_keywords), 1),
            "query_intent": self._analyze_query_intent(query),
        }
