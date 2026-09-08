# 유틸리티 모듈
from .document_processor import DocumentProcessor
from .logging_config import setup_logging, get_logger

# 기본 모듈 (의존성 체크 필요)
try:
    from .text_processing import TextProcessor

    TEXTPROCESSOR_AVAILABLE = True
except ImportError:
    TextProcessor = None
    TEXTPROCESSOR_AVAILABLE = False

# 선택적 import (sklearn 의존성)
try:
    from .keyword_expander import KeywordExpander

    KEYWORDEXPANDER_AVAILABLE = True
except ImportError:
    KeywordExpander = None
    KEYWORDEXPANDER_AVAILABLE = False

# 선택적 import (tiktoken 의존성)
try:
    from .token_counter import TokenCounter

    TOKENCOUNTER_AVAILABLE = True
except ImportError:
    TokenCounter = None
    TOKENCOUNTER_AVAILABLE = False

__all__ = ["DocumentProcessor", "setup_logging", "get_logger"]
if TEXTPROCESSOR_AVAILABLE:
    __all__.append("TextProcessor")
if KEYWORDEXPANDER_AVAILABLE:
    __all__.append("KeywordExpander")
if TOKENCOUNTER_AVAILABLE:
    __all__.append("TokenCounter")
