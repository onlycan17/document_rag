"""
동적 키워드 확장 모듈

사용자 질문에서 핵심 단어를 추출하고 동의어, 유사어, 연관어를 추가하여
검색 성능을 향상시키는 기능을 제공합니다.
"""

from typing import List, Dict, Set, Tuple, Optional
import re
import logging
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.synonym_dict = self._build_enhanced_synonym_dict()
        self.domain_terms = self._build_domain_specific_terms()
        self.stopwords = self._build_korean_stopwords()
        self.embedding_model = embedding_model
        
        # 의미 기반 확장을 위한 캐시
        self.semantic_cache = {}
        
        logger.info("키워드 확장기 초기화 완료")
    
    def _build_enhanced_synonym_dict(self) -> Dict[str, List[str]]:
        """
        향상된 한국어 동의어/유사어 사전 구축
        
        Returns:
            동의어 사전
        """
        return {
            # 정보시스템 관련
            "시스템": ["정보시스템", "전산시스템", "IT시스템", "컴퓨터시스템", "시스템", "체계", "프로그램"],
            "정보시스템": ["시스템", "전산시스템", "IT시스템", "정보화시스템", "정보체계", "DB시스템"],
            "전산": ["컴퓨터", "IT", "정보기술", "전자계산", "디지털", "전산화"],
            "데이터": ["자료", "정보", "데이터베이스", "DB", "정보자료", "전자자료"],
            "데이터베이스": ["DB", "데이터", "자료베이스", "정보저장소", "자료저장"],
            
            # 구축/개발 관련
            "구축": ["건설", "개발", "설치", "도입", "구현", "설립", "조성", "건립"],
            "개발": ["구축", "제작", "작성", "생성", "구현", "설계", "프로그래밍"],
            "설치": ["구축", "설정", "배치", "도입", "세팅", "셋업", "인스톨"],
            "도입": ["구축", "설치", "채택", "적용", "운용", "활용"],
            "구현": ["개발", "구축", "실현", "실행", "완성", "제작"],
            
            # 운영/관리 관련
            "운영": ["관리", "운용", "가동", "실행", "진행", "수행", "administration"],
            "관리": ["운영", "통제", "제어", "유지", "maintenance", "management", "관리감독"],
            "유지보수": ["관리", "운영", "정비", "보수", "maintenance", "점검", "관리"],
            "점검": ["검사", "확인", "조사", "검토", "체크", "모니터링", "감시"],
            "모니터링": ["감시", "관찰", "추적", "점검", "확인", "monitoring"],
            
            # 보안 관련
            "보안": ["보호", "security", "정보보호", "사이버보안", "정보보안", "안전"],
            "보호": ["보안", "방어", "차단", "shield", "protection", "안전"],
            "인증": ["authentication", "확인", "검증", "승인", "로그인"],
            "권한": ["권리", "permission", "접근권한", "사용권한", "authority"],
            "암호화": ["encryption", "보안", "암호", "encoding", "코드화"],
            
            # 법령/규정 관련
            "지침": ["가이드", "가이드라인", "안내서", "매뉴얼", "규정", "규칙", "기준"],
            "규정": ["규칙", "지침", "규범", "법규", "기준", "조례", "지시"],
            "가이드라인": ["지침", "가이드", "안내", "매뉴얼", "지시사항", "기준"],
            "법령": ["법률", "규정", "조례", "법규", "규범", "제도"],
            "정책": ["방침", "지침", "계획", "제도", "규정", "전략"],
            
            # 문서/자료 관련
            "문서": ["서류", "자료", "파일", "문서자료", "기록", "문건"],
            "자료": ["데이터", "정보", "문서", "기록", "material", "문서자료"],
            "기록": ["문서", "자료", "로그", "log", "데이터", "정보"],
            "보고서": ["리포트", "report", "문서", "자료", "결과", "분석"],
            
            # 기관/조직 관련
            "기관": ["기구", "조직", "단체", "회사", "업체", "기업", "organization"],
            "공공기관": ["정부기관", "국가기관", "공기업", "공단", "공사"],
            "행정기관": ["정부기관", "관청", "공공기관", "행정부", "관공서"],
            "정부": ["국가", "행정부", "정부기관", "관청", "공공"],
            
            # 절차/과정 관련
            "절차": ["과정", "단계", "프로세스", "process", "순서", "방법"],
            "과정": ["절차", "단계", "프로세스", "process", "방법", "경로"],
            "단계": ["과정", "절차", "스텝", "step", "phase", "순서"],
            "방법": ["방식", "절차", "과정", "방안", "수단", "기법"],
            "프로세스": ["과정", "절차", "단계", "process", "워크플로우"],
            
            # 계획/전략 관련
            "계획": ["방안", "plan", "기획", "설계", "전략", "스케줄"],
            "방안": ["계획", "방법", "대안", "방식", "solution", "해결책"],
            "전략": ["계획", "방안", "정책", "strategy", "기획", "방침"],
            "기획": ["계획", "설계", "방안", "planning", "전략", "구상"],
            
            # 기술/기능 관련
            "기술": ["기능", "technology", "테크놀로지", "스킬", "능력", "방법"],
            "기능": ["functionality", "능력", "기술", "특성", "성능", "역할"],
            "성능": ["퍼포먼스", "기능", "능력", "효율", "처리능력", "속도"],
            "효율": ["효과", "성능", "생산성", "능률", "최적화"],
            
            # 표준/기준 관련
            "표준": ["기준", "standard", "규격", "규범", "표준안", "가이드"],
            "기준": ["표준", "standard", "규격", "지표", "척도", "기준점"],
            "규격": ["표준", "기준", "specification", "스펙", "규정", "사양"],
            
            # 검사/평가 관련
            "검사": ["점검", "확인", "조사", "검증", "체크", "심사"],
            "평가": ["심사", "검토", "분석", "assessment", "evaluation", "판단"],
            "검토": ["검사", "점검", "확인", "분석", "review", "심사"],
            "분석": ["분석", "해석", "조사", "연구", "검토", "analysis"],
            
            # 업무/작업 관련
            "업무": ["작업", "일", "task", "업무처리", "업무수행", "직무"],
            "작업": ["업무", "일", "task", "work", "처리", "수행"],
            "처리": ["수행", "실행", "진행", "처리작업", "업무", "작업"],
            "수행": ["실행", "처리", "진행", "완수", "업무", "작업"],
            
            # 결과/성과 관련
            "결과": ["성과", "결론", "output", "아웃풋", "산출물", "도출"],
            "성과": ["결과", "성취", "효과", "outcome", "실적", "업적"],
            "효과": ["결과", "성과", "영향", "impact", "효능", "작용"],
            
            # 질문 관련 키워드
            "무엇": ["what", "뭐", "어떤", "어떤것", "어떤거"],
            "어떻게": ["how", "방법", "어떤방법", "어떤식으로", "어떻게하는지"],
            "언제": ["when", "시기", "언제부터", "언제까지", "기간"],
            "어디": ["where", "장소", "위치", "어디서", "어느곳"],
            "왜": ["why", "이유", "원인", "목적", "까닭"],
            "누가": ["who", "누구", "담당자", "책임자", "관리자"],
            
            # 기타 중요 용어들
            "중요": ["핵심", "주요", "필수", "중대", "핵심적", "주된"],
            "필요": ["필수", "요구", "요구사항", "requirement", "조건"],
            "문제": ["이슈", "issue", "문제점", "오류", "error", "장애"],
            "해결": ["solution", "해결책", "방법", "처리", "개선", "문제해결"],
            "개선": ["향상", "발전", "개선사항", "보완", "최적화", "업그레이드"],
            "변경": ["수정", "변화", "개정", "update", "업데이트", "갱신"],
        }
    
    def _build_domain_specific_terms(self) -> Dict[str, List[str]]:
        """
        도메인별 전문 용어 사전 구축
        
        Returns:
            도메인별 용어 사전
        """
        return {
            "it": ["시스템", "정보", "데이터", "서버", "네트워크", "소프트웨어", "하드웨어", "프로그램"],
            "security": ["보안", "암호화", "인증", "권한", "방화벽", "접근제어", "보호"],
            "management": ["관리", "운영", "정책", "계획", "전략", "조직", "업무"],
            "government": ["행정", "정부", "공공", "기관", "법령", "규정", "정책"],
            "development": ["개발", "구축", "설계", "구현", "프로그래밍", "시스템개발"],
            "quality": ["품질", "표준", "기준", "평가", "검사", "인증", "규격"],
        }
    
    def _build_korean_stopwords(self) -> Set[str]:
        """
        한국어 불용어 세트 구축
        
        Returns:
            불용어 세트
        """
        return {
            # 조사
            '이', '그', '저', '의', '가', '을', '를', '에', '와', '과', '도', '로', '으로', 
            '는', '은', '에서', '부터', '까지', '보다', '처럼', '같이', '마다', '마저', '조차',
            
            # 어미
            '이다', '있다', '없다', '하다', '되다', '시키다', '당하다', '받다', '주다',
            
            # 대명사
            '이것', '그것', '저것', '여기', '거기', '저기', '이곳', '그곳', '저곳',
            
            # 수사
            '하나', '둘', '셋', '넷', '다섯', '한', '두', '세', '네', '다섯',
            
            # 관형사
            '모든', '모든지', '어떤', '어느', '무슨', '어떤', '다른',
            
            # 부사
            '매우', '아주', '정말', '너무', '많이', '조금', '약간', '대단히', '상당히',
            '또한', '또는', '그리고', '하지만', '그러나', '따라서', '그래서', '즉',
            
            # 감탄사
            '아', '어', '오', '우', '음', '으', '헉', '와', '우와',
            
            # 기타
            '수', '것', '등', '및', '또는', '또한', '즉', '만', '제', '위', '때', '중',
            '들', '말', '점', '곳', '바', '듯', '채', '편', '쪽', '번', '개', '명',
        }
    
    def expand_keywords(self, query: str, max_terms: int = 25, 
                       use_semantic: bool = True, use_domain: bool = True) -> List[str]:
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
            final_keywords = self._rank_and_limit_keywords(
                all_keywords, query, max_terms
            )
            
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
        cleaned_query = re.sub(r'[^\w\s가-힣]', ' ', query)
        words = cleaned_query.split()
        
        # 불용어 제거 및 길이 필터링
        core_keywords = []
        for word in words:
            word = word.strip()
            if (len(word) > 1 and 
                word not in self.stopwords and
                not word.isdigit()):
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
            "who": ["담당자", "책임자", "관리자", "주체", "기관", "누구", "담당"]
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
                for keyword2 in keywords[i+1:]:
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
        
        return list(set(contextual_keywords))[:8]  # 중복 제거 후 최대 8개
    
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
            "who": ["누가", "누구", "담당", "책임"]
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
            if (keyword not in seen and 
                len(keyword) > 1 and 
                keyword not in self.stopwords):
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _rank_and_limit_keywords(self, keywords: List[str], 
                                original_query: str, max_terms: int) -> List[str]:
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
            korean_ratio = sum(1 for c in keyword if '\uac00' <= c <= '\ud7a3') / len(keyword)
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
            "query_intent": self._analyze_query_intent(query)
        } 