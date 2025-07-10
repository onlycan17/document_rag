"""프로젝트 전체에서 사용되는 상수 정의"""
from typing import Dict, Set, Any

# 파일 경로 패턴
LOG_FILE_PATTERN = "./logs/rag_app_{date}.log"
TEMP_DOCUMENT_PATH = "./data/documents/{filename}"
VECTOR_DB_FAISS_INDEX = "faiss_index.pkl"

# UI 관련 상수
MAX_LOG_LINES_DISPLAY = 50
PROGRESS_UPDATE_INTERVAL = 0.05  # 5%

# 검색 관련 상수
DEFAULT_MMR_DIVERSITY = 0.3
MODEL_SAFETY_MARGIN = 0.7  # 기본 모델 안전 마진
LOCAL_MODEL_SAFETY_MARGIN = 0.5  # 로컬 모델 안전 마진
MIN_SEARCH_RESULTS = 2  # 필터링 후 최소 결과 수
MIN_RELEVANCE_SCORE = 0.1  # TF-IDF 최소 관련성 점수

# 텍스트 처리 관련 상수
MIN_TEXT_LENGTH = 10
MAX_QUERY_TERMS = 30
MAX_CONTEXT_TERMS = 20
DEFAULT_TOP_KEYWORDS = 10

# 모델별 설정
MODEL_PROMPT_TOKENS = {
    "default": 800,
    "local": 600
}

MODEL_MIN_CONTEXT = {
    "default": 4000,
    "local": 2000
}

MODEL_MAX_CONTEXT = {
    "default": 500000,
    "local": 6000,
    "gpt-3.5": 30000
}

# 문서 처리 관련
OPTIMAL_DOC_LENGTH_RANGE = (200, 1000)  # 문서 최적 길이 범위
MAX_DOC_LENGTH_SCORE = 2000  # 이 길이를 초과하면 점수 감소

# 토큰 관련 상수
TOKEN_TO_CHAR_RATIO = 4  # 평균적으로 1토큰 = 4문자

# 벡터 검색 가중치
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3

# 파일 크기 제한 (MB)
MAX_FILE_SIZE_MB = 100
LARGE_FILE_WARNING_MB = 50

# 청크 관련 기본값
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
SEMANTIC_CHUNK_SENTENCES = 3

# 검색 결과 상수
MAX_SEARCH_RESULTS = 20
DEFAULT_K_DOCUMENTS = 8
LOCAL_MODEL_MAX_DOCUMENTS = 4

# TF-IDF 설정
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1

# 재시도 및 타임아웃 설정
REQUEST_TIMEOUT = 2  # 초
MAX_RETRIES = 3
RETRY_DELAY = 1  # 초