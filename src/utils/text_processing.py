"""텍스트 처리 관련 공통 유틸리티"""

from typing import List, Set, Dict
import re
import logging
import unicodedata
from .keyword_expander import KeywordExpander

logger = logging.getLogger(__name__)


class TextProcessor:
    """텍스트 전처리 및 쿼리 확장을 위한 공통 유틸리티"""

    # 클래스 변수로 KeywordExpander 인스턴스
    _keyword_expander = None

    # 한국어 불용어 세트
    KOREAN_STOPWORDS: Set[str] = {
        "이",
        "그",
        "저",
        "의",
        "가",
        "을",
        "를",
        "에",
        "와",
        "과",
        "도",
        "로",
        "으로",
        "는",
        "은",
        "이다",
        "있다",
        "없다",
        "하다",
        "되다",
        "수",
        "것",
        "등",
        "및",
        "또는",
        "또한",
        "즉",
        "만",
        "제",
        "위",
    }

    @classmethod
    def get_keyword_expander(cls, embedding_model=None) -> KeywordExpander:
        """
        KeywordExpander 인스턴스를 반환 (싱글톤 패턴)

        Args:
            embedding_model: 임베딩 모델 인스턴스 (옵션)

        Returns:
            KeywordExpander 인스턴스
        """
        if cls._keyword_expander is None:
            cls._keyword_expander = KeywordExpander(embedding_model=embedding_model)
        elif embedding_model and cls._keyword_expander.embedding_model is None:
            # 임베딩 모델이 새로 제공되었지만 기존 인스턴스에는 없는 경우
            cls._keyword_expander.embedding_model = embedding_model
        return cls._keyword_expander

    # 쿼리 확장 사전
    QUERY_EXPANSIONS: Dict[str, str] = {
        # 몽촌토성 관련
        "몽촌토성": "몽촌토성 몽촌 토성 백제 한성 왕성 토성 백제왕성 백제토성 한성백제토성",
        "몽촌": "몽촌 몽촌토성 백제 한성",
        "토성": "토성 몽촌토성 성곽 성벽 토축성 판축",
        # 백제/고구려 관련
        "백제": "백제 한성백제 백제시대 백제왕조 백제왕국 백제토기",
        "고구려": "고구려 고구려시대 고구려토기 고구려유물",
        "한성": "한성 한성백제 한성시대 한성도읍 서울",
        # 고고학 관련
        "발굴": "발굴 발굴조사 고고학 유적 출토 조사 시굴 정밀발굴",
        "유물": "유물 토기 유구 출토품 출토유물 도자기 자기",
        "토기": "토기 도기 자기 그릇 토제품 백제토기 고구려토기",
        "유적": "유적 유구 유물 흔적 건물지 주거지",
        # 지역 관련
        "북문": "북문 북문지 북쪽문 북측",
        "남문": "남문 남문지 남쪽문 남측",
        "동문": "동문 동문지 동쪽문 동측",
        "서문": "서문 서문지 서쪽문 서측",
        # 시대 관련
        "삼국시대": "삼국시대 백제 고구려 신라 삼국",
        "통일신라": "통일신라 통일신라시대 신라",
        # 정보시스템 관련
        "시스템": "정보시스템 시스템 전산시스템 IT시스템",
        "구축": "구축 건설 개발 설치 도입 구현",
        "운영": "운영 관리 유지보수 운용 administration",
        "지침": "지침 가이드 규정 가이드라인 매뉴얼 안내서",
        "보안": "보안 security 정보보호 보호 사이버보안",
        "관리": "관리 management 관리자 운영관리 시스템관리",
    }

    @staticmethod
    def clean_text(text: str, remove_stopwords: bool = False) -> str:
        """
        텍스트 정제

        Args:
            text: 정제할 텍스트
            remove_stopwords: 불용어 제거 여부

        Returns:
            정제된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""

        # 공백 정규화
        text = " ".join(text.split())

        # 특수 문자 정리
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\xa0", " ")  # Non-breaking space

        # 연속된 특수문자 제거
        text = re.sub(r"[^\w\s가-힣a-zA-Z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        # 불용어 제거 (선택적)
        if remove_stopwords:
            words = text.split()
            filtered_words = [word for word in words if word not in TextProcessor.KOREAN_STOPWORDS and len(word) > 1]
            text = " ".join(filtered_words)

        return text.strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """공백 정규화"""
        return " ".join(text.split())

    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = False) -> str:
        """특수문자 제거"""
        if keep_punctuation:
            # 구두점은 유지하면서 특수문자 제거
            pattern = r"[^\w\s가-힣a-zA-Z0-9.,!?;:\-\(\)]"
        else:
            # 모든 특수문자 제거
            pattern = r"[^\w\s가-힣a-zA-Z0-9]"

        return re.sub(pattern, " ", text)

    @staticmethod
    def expand_query(
        query: str, use_static_expansion: bool = True, use_dynamic_expansion: bool = True, max_terms: int = 25
    ) -> str:
        """
        향상된 쿼리 확장

        Args:
            query: 확장할 쿼리
            use_static_expansion: 정적 확장 사전 사용 여부 (하위 호환성)
            use_dynamic_expansion: 동적 키워드 확장 사용 여부
            max_terms: 최대 키워드 개수

        Returns:
            확장된 쿼리
        """
        if not query:
            return ""

        # 동적 키워드 확장 사용
        if use_dynamic_expansion:
            try:
                keyword_expander = TextProcessor.get_keyword_expander()
                expanded_keywords = keyword_expander.expand_keywords(
                    query, max_terms=max_terms, use_semantic=True, use_domain=True
                )

                if expanded_keywords:
                    result = " ".join(expanded_keywords)
                    logger.info(f"동적 키워드 확장: '{query}' -> '{result[:100]}...'")
                    return result

            except Exception as e:
                logger.warning(f"동적 키워드 확장 실패, 정적 확장 사용: {str(e)}")

        # 기존 정적 확장 로직 (하위 호환성)
        if use_static_expansion:
            expanded_terms = []
            query_words = query.split()

            for word in query_words:
                if word in TextProcessor.QUERY_EXPANSIONS:
                    expanded_terms.append(TextProcessor.QUERY_EXPANSIONS[word])
                else:
                    # 부분 일치 확인
                    for key, value in TextProcessor.QUERY_EXPANSIONS.items():
                        if key in word or word in key:
                            expanded_terms.append(value)
                            break
                    else:
                        expanded_terms.append(word)

            # 중복 제거하면서 확장된 쿼리 생성
            all_terms = []
            for term in expanded_terms:
                if isinstance(term, str):
                    all_terms.extend(term.split())

            # 중복 제거 (순서 유지)
            seen = set()
            unique_terms = []
            for term in all_terms:
                if term not in seen and len(term) > 1:
                    seen.add(term)
                    unique_terms.append(term)

            return " ".join(unique_terms[:max_terms])

        # 확장 없이 원본 반환
        return query

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """
        텍스트에서 주요 키워드 추출

        Args:
            text: 키워드를 추출할 텍스트
            top_k: 추출할 키워드 개수

        Returns:
            추출된 키워드 리스트
        """
        # 텍스트 정제
        cleaned_text = TextProcessor.clean_text(text, remove_stopwords=True)

        # 단어 빈도 계산
        words = cleaned_text.split()
        word_freq = {}

        for word in words:
            if len(word) > 1:  # 1글자 단어 제외
                word_freq[word] = word_freq.get(word, 0) + 1

        # 빈도순 정렬
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # 상위 k개 추출
        return [word for word, freq in sorted_words[:top_k]]

    @staticmethod
    def is_valid_text(text: str, min_length: int = 10) -> bool:
        """
        텍스트 유효성 검사

        Args:
            text: 검사할 텍스트
            min_length: 최소 길이

        Returns:
            유효 여부
        """
        if not text or not isinstance(text, str):
            return False

        cleaned = TextProcessor.clean_text(text)
        return len(cleaned) >= min_length

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        파일명 안전화 및 한글 정규화 (macOS NFD 문제 방지)
        - Unicode를 NFC로 정규화하여 자모 분리 현상 방지
        - 제어 문자, 위험 문자를 제거
        - 공백은 그대로 두되, 연속 공백은 하나로 축소
        """
        if not name:
            return ""
        # 유니코드 정규화 (NFC)
        name = unicodedata.normalize("NFC", name)
        # 불필요한 제어 문자 제거
        name = re.sub(r"[\x00-\x1F\x7F]", "", name)
        # 파일명에 부적합한 문자 제거 (경로 구분자, 일부 특수문자)
        name = re.sub(r"[\\/:*?\"<>|]", "_", name)
        # 연속 공백 축소
        name = re.sub(r"\s+", " ", name).strip()
        return name
