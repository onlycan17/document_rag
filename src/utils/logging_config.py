import logging
import sys
from datetime import datetime
import os


def setup_logging(log_level: int = logging.INFO):
    """로깅 설정 (강제 파일 핸들러 부착)

    일부 프레임워크(Streamlit 등)가 이미 로깅을 초기화해둔 경우
    logging.basicConfig가 무시될 수 있어, 루트 로거에 파일/스트림 핸들러를
    직접 점검 후 추가합니다.
    """
    # 로그 디렉토리 생성
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일명 (날짜별)
    log_file = os.path.join(log_dir, f"rag_app_{datetime.now().strftime('%Y%m%d')}.log")

    # 포맷터
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 루트 로거 가져오기 및 레벨 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 파일 핸들러가 이미 추가되어 있는지 확인 (중복 방지)
    file_handler_exists = False
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                # 동일 파일로 기록 중인 핸들러가 있으면서 레벨/포맷이 다른 경우 교체
                same_file = os.path.abspath(getattr(h, "baseFilename", "")) == os.path.abspath(log_file)
                if same_file:
                    file_handler_exists = True
                    # 포맷터/레벨 정합성 점검
                    if getattr(h, "level", None) != log_level or getattr(h, "formatter", None) != logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    ):
                        root_logger.removeHandler(h)
                        file_handler_exists = False
                else:
                    # 다른 로그 파일 핸들러는 유지
                    pass
            except Exception as err:
                logging.getLogger(__name__).debug(f"파일 핸들러 점검 실패(무시): {err}")
                continue

    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 스트림 핸들러(표준 출력) 존재 여부 확인 후 추가 (중복 방지)
    stream_handler_exists = False
    for h in list(root_logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            stream_handler_exists = True
            # 레벨/포맷 불일치 시 갱신
            try:
                h.setLevel(log_level)
                h.setFormatter(formatter)
            except Exception as err:
                logging.getLogger(__name__).debug(f"스트림 핸들러 갱신 실패(무시): {err}")
            break
    if not stream_handler_exists:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    # 시끄러운 서드파티 로거 억제
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def get_logger(name):
    """로거 인스턴스 반환"""
    return logging.getLogger(name)
