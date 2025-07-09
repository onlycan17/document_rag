import logging
import sys
from datetime import datetime
import os

def setup_logging(log_level=logging.INFO):
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 (날짜별)
    log_file = os.path.join(log_dir, f"rag_app_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 로깅 포맷
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 루트 로거 설정
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 특정 로거의 레벨 조정
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def get_logger(name):
    """로거 인스턴스 반환"""
    return logging.getLogger(name)