"""
RAG 챗봇 메인 애플리케이션

Streamlit 기반의 RAG(Retrieval-Augmented Generation) 챗봇 웹 애플리케이션입니다.
문서 업로드, 벡터 데이터베이스 관리, 다양한 LLM 모델 지원 등의 기능을 제공합니다.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st

# ChromaDB 텔레메트리 비활성화 (가장 먼저 실행)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# PyTorch torch.classes 경고 메시지 전역 억제 (A.X 모델용) - 강화된 버전
os.environ["TORCH_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_KERNEL_WARN"] = "0"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

import warnings
import logging

# 전체 PyTorch 및 transformers 경고 억제
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*torch.ops.*")
warnings.filterwarnings("ignore", message=".*torch.jit.*")
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ops.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*torch.*")

# PyTorch 관련 로거 억제
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.jit").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)
logging.getLogger("torch._C").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Streamlit에서 torch.classes 경고를 원천 차단
try:
    import torch
    import torch._C
    # JIT 경고 완전 비활성화
    if hasattr(torch._C, '_jit_set_emit_warnings'):
        torch._C._jit_set_emit_warnings(False)
    if hasattr(torch._C, '_set_print_stacktraces_on_fatal_signal'):
        torch._C._set_print_stacktraces_on_fatal_signal(False)
    
    # torch.classes 관련 내부 경고 시스템 비활성화
    if hasattr(torch, '_C') and hasattr(torch._C, '_set_print_warn'):
        try:
            torch._C._set_print_warn(False)
        except:
            pass
            
except Exception:
    pass

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from config import settings
from src.loaders import DocumentLoader
from src.rag import RAGChain
from src.vectorstore import VectorDatabase
from src.utils.logging_config import setup_logging, get_logger
from src.utils.token_counter import TokenCounter
from src.utils.document_processor import DocumentProcessor
from src.constants import (
    LOG_FILE_PATTERN, TEMP_DOCUMENT_PATH, MAX_LOG_LINES_DISPLAY
)

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)

# 모델 부트스트랩은 세션 상태에서 한 번만 실행
# (모듈 레벨 실행으로 인한 중복 방지)

# 페이지 설정
st.set_page_config(
    page_title=settings.app_title,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': None,
        'Get Help': None,
        'About': settings.app_description
    }
)

# 세션 상태 안전 접근 헬퍼 함수들
def safe_get_vector_db():
    """벡터 DB 안전 접근 - 초기화되지 않은 경우 자동 초기화 (중복 방지)"""
    if 'vector_db' not in st.session_state or st.session_state.vector_db is None:
        # 중복 초기화 방지를 위한 플래그 확인
        if not st.session_state.get('vector_db_initializing', False):
            st.session_state.vector_db_initializing = True
            logger.info("🔄 vector_db 초기화 시작 (한 번만)")
            try:
                st.session_state.vector_db = VectorDatabase()
                logger.info("✅ vector_db 초기화 완료")
            finally:
                st.session_state.vector_db_initializing = False
        else:
            # 다른 프로세스에서 초기화 중인 경우 잠시 대기
            import time
            for _ in range(10):  # 최대 1초 대기
                if 'vector_db' in st.session_state and st.session_state.vector_db is not None:
                    break
                time.sleep(0.1)
            
            # 여전히 없으면 강제 초기화
            if 'vector_db' not in st.session_state or st.session_state.vector_db is None:
                logger.warning("⚠️ vector_db 대기 시간 초과 - 강제 초기화")
                st.session_state.vector_db = VectorDatabase()
    
    return st.session_state.vector_db

def safe_get_rag_chain():
    """RAG 체인 안전 접근 - 초기화되지 않은 경우 자동 초기화 (중복 방지)"""
    if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
        # 중복 초기화 방지를 위한 플래그 확인
        if not st.session_state.get('rag_chain_initializing', False):
            st.session_state.rag_chain_initializing = True
            logger.info("🔄 rag_chain 초기화 시작 (한 번만)")
            try:
                vector_db = safe_get_vector_db()
                st.session_state.rag_chain = RAGChain(vector_db=vector_db)
                logger.info("✅ rag_chain 초기화 완료")
            finally:
                st.session_state.rag_chain_initializing = False
        else:
            # 다른 프로세스에서 초기화 중인 경우 잠시 대기
            import time
            for _ in range(10):  # 최대 1초 대기
                if 'rag_chain' in st.session_state and st.session_state.rag_chain is not None:
                    break
                time.sleep(0.1)
            
            # 여전히 없으면 강제 초기화
            if 'rag_chain' not in st.session_state or st.session_state.rag_chain is None:
                logger.warning("⚠️ rag_chain 대기 시간 초과 - 강제 초기화")
                vector_db = safe_get_vector_db()
                st.session_state.rag_chain = RAGChain(vector_db=vector_db)
    
    return st.session_state.rag_chain

# 세션 상태 초기화
# 모델 부트스트랩을 프로세스 간 동기화로 한 번만 실행
def bootstrap_models_with_file_lock():
    """파일 기반 잠금을 사용한 멀티프로세스 안전 모델 부트스트랩"""
    import fcntl
    import time
    
    # 잠금 파일 경로
    lock_file_path = Path(__file__).parent / ".bootstrap_lock"
    bootstrap_done_file = Path(__file__).parent / ".bootstrap_done"
    
    # 이미 부트스트랩이 완료되었다면 스킷
    if bootstrap_done_file.exists():
        logger.debug("모델 부트스트랩이 이미 완료됨 (파일 확인)")
        return True
    
    # 잠금 파일로 동기화
    try:
        with open(lock_file_path, 'w') as lock_file:
            try:
                # 비블로킹 잠금 시도
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # 잠금 획득 성공 - 부트스트랩 실행
                if not bootstrap_done_file.exists():
                    logger.info("🚀 프로세스 동기화 모델 부트스트랩 시작...")
                    
                    from src.utils.model_bootstrap import ensure_models_available, preload_models
                    ensure_models_available(download_exaone=False)
                    preload_models()
                    
                    # 완료 표시 파일 생성
                    bootstrap_done_file.touch()
                    logger.info("✅ 프로세스 동기화 모델 부트스트랩 완료")
                    
                return True
                
            except BlockingIOError:
                # 다른 프로세스가 이미 잠금 보유 중 - 대기
                logger.debug("다른 프로세스가 부트스트랩 중... 대기")
                
                # 최대 30초 대기
                max_wait = 30
                wait_time = 0
                while not bootstrap_done_file.exists() and wait_time < max_wait:
                    time.sleep(0.5)
                    wait_time += 0.5
                
                if bootstrap_done_file.exists():
                    logger.debug("다른 프로세스의 부트스트랩 완료 확인")
                    return True
                else:
                    logger.warning("부트스트랩 대기 시간 초과")
                    return False
                    
    except Exception as e:
        logger.warning(f"파일 잠금 부트스트랩 실패: {e}")
        return False
    finally:
        # 잠금 파일 정리 (완료 파일은 유지)
        try:
            if lock_file_path.exists():
                lock_file_path.unlink()
        except:
            pass

# 프로세스 동기화 부트스트랩 실행
if 'models_bootstrapped' not in st.session_state:
    try:
        success = bootstrap_models_with_file_lock()
        st.session_state.models_bootstrapped = success
        if not success:
            logger.warning("프로세스 동기화 부트스트랩 실패 - 폴백 시도")
            # 폴백: 기존 방식 시도
            from src.utils.model_bootstrap import ensure_models_available, preload_models
            ensure_models_available(download_exaone=False)
            preload_models()
            st.session_state.models_bootstrapped = True
    except Exception as e:
        logger.warning(f"모델 부트스트랩 중 경고: {e}")
        st.session_state.models_bootstrapped = False

# 전역 초기화 상태를 프로세스 간 동기화
def initialize_components_with_sync():
    """프로세스 간 동기화된 컴포넌트 초기화"""
    import fcntl
    import time
    
    # 컴포넌트 초기화 잠금 파일
    comp_lock_file = Path(__file__).parent / ".components_lock"
    comp_done_file = Path(__file__).parent / ".components_done"
    
    # 이미 초기화 완료되었다면 스킵
    if comp_done_file.exists():
        logger.debug("컴포넌트 초기화가 이미 완료됨")
        return True
    
    try:
        with open(comp_lock_file, 'w') as lock_file:
            try:
                # 비블로킹 잠금 시도
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # 잠금 획득 성공 - 첫 번째 프로세스가 초기화 수행
                if not comp_done_file.exists():
                    logger.info("🔧 프로세스 동기화 컴포넌트 초기화 시작...")
                    
                    # VectorDatabase 초기화 (로그 중복 방지를 위해 한 번만)
                    if 'vector_db' not in st.session_state:
                        st.session_state.vector_db = VectorDatabase()
                    
                    # RAG 체인 초기화
                    if 'rag_chain' not in st.session_state:
                        st.session_state.rag_chain = RAGChain(vector_db=safe_get_vector_db())
                    
                    # 초기화 완료 표시
                    comp_done_file.touch()
                    logger.info("✅ 프로세스 동기화 컴포넌트 초기화 완료")
                
                return True
                
            except BlockingIOError:
                # 다른 프로세스가 초기화 중 - 대기
                logger.debug("다른 프로세스가 컴포넌트 초기화 중... 대기")
                
                # 최대 20초 대기
                max_wait = 20
                wait_time = 0
                while not comp_done_file.exists() and wait_time < max_wait:
                    time.sleep(0.3)
                    wait_time += 0.3
                
                if comp_done_file.exists():
                    logger.debug("다른 프로세스의 컴포넌트 초기화 완료 확인")
                    # 여전히 세션 상태에는 설정해야 함
                    if 'vector_db' not in st.session_state:
                        st.session_state.vector_db = VectorDatabase()
                    if 'rag_chain' not in st.session_state:
                        st.session_state.rag_chain = RAGChain(vector_db=safe_get_vector_db())
                    return True
                else:
                    logger.warning("컴포넌트 초기화 대기 시간 초과")
                    return False
                    
    except Exception as e:
        logger.warning(f"컴포넌트 동기화 실패: {e}")
        return False
    finally:
        # 잠금 파일 정리
        try:
            if comp_lock_file.exists():
                comp_lock_file.unlink()
        except:
            pass

# 동기화된 컴포넌트 초기화 실행
if 'vector_db' not in st.session_state or 'rag_chain' not in st.session_state:
    try:
        success = initialize_components_with_sync()
        if not success:
            logger.warning("동기화 컴포넌트 초기화 실패 - 폴백 실행")
            # 폴백: 개별 초기화
            if 'vector_db' not in st.session_state:
                st.session_state.vector_db = VectorDatabase()
            if 'rag_chain' not in st.session_state:
                st.session_state.rag_chain = RAGChain(vector_db=safe_get_vector_db())
    except Exception as e:
        logger.warning(f"컴포넌트 초기화 중 오류: {e}")
        # 폴백 실행
        if 'vector_db' not in st.session_state:
            st.session_state.vector_db = VectorDatabase()
        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = RAGChain(vector_db=safe_get_vector_db())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_loader' not in st.session_state:
    st.session_state.document_loader = DocumentLoader(
        use_ocr=True, 
        use_agent_preprocessing=False, 
        enable_postprocessing=False,
        use_intelligent_image_extraction=False
    )
if 'current_provider' not in st.session_state:
    st.session_state.current_provider = settings.llm_provider
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'token_counter' not in st.session_state:
    st.session_state.token_counter = TokenCounter()

def analyze_chunks(documents: List[Any]) -> Optional[Dict[str, Any]]:
    """
    업로드된 문서의 청크 분석
    
    Args:
        documents: 분석할 문서 청크 리스트
        
    Returns:
        청크 분석 결과 딕셔너리 또는 None (문서가 없는 경우)
        - count: 청크 개수
        - avg_size: 평균 크기
        - min_size: 최소 크기
        - max_size: 최대 크기
        - total_chars: 총 문자 수
    """
    if not documents:
        return None
    
    chunk_sizes = [len(doc.page_content) for doc in documents]
    return {
        'count': len(documents),
        'avg_size': sum(chunk_sizes) / len(chunk_sizes),
        'min_size': min(chunk_sizes),
        'max_size': max(chunk_sizes),
        'total_chars': sum(chunk_sizes)
    }

def get_pdf_metadata(documents: List[Any]) -> Optional[Dict[str, Any]]:
    """
    PDF 메타데이터 추출
    
    Args:
        documents: 문서 리스트 (첫 번째 문서의 메타데이터 사용)
        
    Returns:
        PDF 메타데이터 딕셔너리 또는 None (문서가 없는 경우)
        - extraction_method: 추출 방법
        - page_count: 페이지 수
        - ocr_language: OCR 언어
        - processing_method: 처리 방법
        - image_count: 이미지 개수
        - conversion_status: 변환 상태
    """
    if not documents:
        return None
    
    metadata = documents[0].metadata
    return {
        'extraction_method': metadata.get('extraction_method', metadata.get('processing_method', 'standard')),
        'page_count': metadata.get('page_count', 0),
        'ocr_language': metadata.get('ocr_language', None),
        'processing_method': metadata.get('processing_method', 'standard'),
        'image_count': metadata.get('image_count', 0),
        'conversion_status': metadata.get('conversion_status', 'unknown')
    }

def test_search_quality(vector_db: Any, filename: str) -> Dict[str, Any]:
    """
    업로드된 문서의 검색 품질 테스트
    
    파일명에서 추출한 키워드로 벡터 데이터베이스 검색을 테스트하여
    문서가 제대로 인덱싱되었는지 확인합니다.
    
    Args:
        vector_db: 벡터 데이터베이스 인스턴스
        filename: 테스트할 파일명
        
    Returns:
        키워드별 검색 결과 개수 딕셔너리
    """
    # 파일명에서 키워드 추출하여 테스트
    test_keywords = []
    filename_lower = filename.lower()
    
    # 한국 관련 키워드
    korean_keywords = ['몽촌토성', '백제', '고고학', '발굴', '유물', '토성', '왕성', '조사']
    for keyword in korean_keywords:
        if keyword in filename_lower:
            test_keywords.append(keyword)
    
    # 기본 키워드 추가
    if not test_keywords:
        test_keywords = ['문서', '내용', '정보']
    
    # 최대 3개 키워드만 테스트
    test_keywords = test_keywords[:3]
    
    search_results = {}
    for keyword in test_keywords:
        try:
            results = vector_db.search(keyword, k=3)
            search_results[keyword] = len(results)
        except Exception as e:
            search_results[keyword] = f"오류: {str(e)}"
    
    return search_results

def serve_image(image_path: str) -> str:
    """
    이미지 파일을 base64로 인코딩하여 Streamlit에서 표시할 수 있는 형태로 변환
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        base64 인코딩된 이미지 데이터 URL
    """
    import base64
    
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                
                # 파일 확장자로 MIME 타입 결정
                ext = Path(image_path).suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext == '.bmp':
                    mime_type = 'image/bmp'
                else:
                    mime_type = 'image/png'  # 기본값
                
                return f"data:{mime_type};base64,{encoded_string}"
        else:
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
    except Exception as e:
        logger.error(f"이미지 서빙 실패: {str(e)}")
        return None

def display_images_in_response(response_text: str, context_documents: List = None) -> str:
    """
    응답 텍스트와 컨텍스트 문서에서 이미지를 찾아 표시
    
    Args:
        response_text: RAG 응답 텍스트
        context_documents: 검색된 컨텍스트 문서들
        
    Returns:
        이미지가 포함된 HTML 마크업이 추가된 응답 텍스트
    """
    if not context_documents:
        return response_text
    
    # 관련 이미지 수집
    related_images = []
    
    for doc in context_documents:
        # doc가 dict인지 Document 객체인지 확인
        if isinstance(doc, dict):
            # sources에서 온 dict 데이터
            images = doc.get('images', [])
            source_name = doc.get('file_name', 'Unknown')
        else:
            # Document 객체에서 온 데이터
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            images = metadata.get('images', [])
            source_name = metadata.get('file_name', 'Unknown')
        
        for image_info in images:
            image_path = image_info.get('path', '')
            filename = image_info.get('filename', 'Unknown')
            
            # PDF 파일 제외 및 실제 이미지 파일만 허용
            if os.path.exists(image_path):
                # 파일 확장자 확인
                _, ext = os.path.splitext(filename.lower())
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    related_images.append({
                        'path': image_path,
                        'filename': filename,
                        'source': source_name
                    })
    
    # 이미지가 있으면 응답에 추가
    if related_images:
        st.markdown("### 📸 관련 이미지")
        
        # PNG 이미지를 우선적으로 정렬
        def image_priority(img_info):
            filename = img_info['filename'].lower()
            if filename.endswith('.png'):
                return 0  # PNG 최우선
            elif filename.endswith(('.jpg', '.jpeg')):
                return 1  # JPEG 두 번째
            else:
                return 2  # 기타 이미지
        
        related_images.sort(key=image_priority)
        
        # 라이트박스용 CSS와 JavaScript 추가
        st.markdown("""
        <style>
        /* 이미지 컨테이너 스타일 - 가장 강력한 선택자 사용 */
        div[onclick*="openLightbox"],
        .stMarkdown div[onclick*="openLightbox"],
        [data-testid="stMarkdownContainer"] div[onclick*="openLightbox"],
        .image-container {
            width: 300px !important;
            height: 240px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border: 2px solid #ddd !important;
            border-radius: 8px !important;
            margin: 0 auto 8px auto !important;
            overflow: hidden !important;
            background-color: #f8f9fa !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-sizing: border-box !important;
        }
        .image-container:hover {
            border-color: #1f77b4 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            transform: scale(1.02) !important;
        }
        .image-thumbnail {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }
        /* Streamlit 기본 이미지 스타일 오버라이드 - 더 강력한 선택자 */
        div.image-container img,
        .image-container img,
        .stMarkdown .image-container img,
        [data-testid="stMarkdownContainer"] .image-container img {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            display: block !important;
        }
        /* 컨테이너 크기 강제 적용 */
        div.image-container,
        .stMarkdown .image-container,
        [data-testid="stMarkdownContainer"] .image-container {
            width: 300px !important;
            height: 240px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            box-sizing: border-box !important;
        }
        .lightbox {
            display: none;
            position: fixed;
            z-index: 999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .lightbox.show {
            display: block;
            opacity: 1;
        }
        .lightbox-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }
        .lightbox-image {
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }
        .lightbox-close {
            position: absolute;
            top: 20px;
            right: 35px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
            transition: color 0.3s;
        }
        .lightbox-close:hover {
            color: #ccc;
        }
        .lightbox-info {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            text-align: center;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 14px;
        }
        /* 사이드바 파일 업로더 레이아웃 수정 */
        .sidebar .stFileUploader > div > div > div {
            padding: 8px 12px !important;
        }
        .sidebar .stFileUploader [data-testid="stFileUploaderDropzone"] {
            min-height: 100px !important;
            padding: 20px 10px !important;
        }
        .sidebar .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
            font-size: 12px !important;
            text-align: center !important;
            line-height: 1.4 !important;
            margin: 0 !important;
            word-break: keep-all !important;
        }
        /* 업로드 버튼과 텍스트 정렬 */
        .sidebar .stFileUploader button {
            font-size: 12px !important;
            padding: 4px 8px !important;
        }
        .sidebar .stFileUploader small {
            font-size: 11px !important;
            line-height: 1.3 !important;
        }
        /* 추가 이미지 크기 제어 */
        .image-container * {
            max-width: 100% !important;
            max-height: 100% !important;
        }
        /* Streamlit 컬럼 내 이미지 컨테이너 */
        .element-container .image-container {
            width: 300px !important;
            height: 240px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 이미지를 열로 나누어 표시 (최대 2개씩으로 조정 - 더 큰 이미지에 맞게)
        cols = st.columns(min(2, len(related_images)))
        
        for i, image_info in enumerate(related_images[:6]):  # 최대 6개까지만 표시
            col_idx = i % 2
            
            with cols[col_idx]:
                image_data_url = serve_image(image_info['path'])
                if image_data_url:
                    # 고유한 ID 생성
                    image_id = f"img_{i}_{hash(image_info['filename']) % 10000}"
                    
                    # 파일명에서 불필요한 부분 제거하고 깔끔하게 표시
                    clean_filename = image_info['filename']
                    if '+' in clean_filename:
                        clean_filename = clean_filename.replace('+', ' ')
                    if '_page' in clean_filename:
                        # _page002_img031.png -> img031.png
                        parts = clean_filename.split('_')
                        if len(parts) > 2:
                            clean_filename = '_'.join(parts[-2:])  # 마지막 두 부분만 유지
                    
                    # 소스 파일명도 깔끔하게
                    clean_source = image_info['source']
                    if '.pdf' in clean_source:
                        clean_source = clean_source.replace('.pdf', '').replace('+', ' ')
                        if len(clean_source) > 30:
                            clean_source = clean_source[:27] + "..."
                    
                    st.markdown(
                        f'''<div class="image-container" data-img-src="{image_data_url}" data-filename="{clean_filename}" data-source="{clean_source}" title="클릭하여 크게 보기">
                            <img id="{image_id}" 
                                src="{image_data_url}" 
                                class="image-thumbnail">
                        </div>''',
                        unsafe_allow_html=True
                    )
                    # 이미지 설명 및 출처 캡션 표시 (메타데이터에 description이 있으면 함께 표시)
                    desc = image_info.get('description') if isinstance(image_info, dict) else None
                    if desc:
                        st.caption(desc)
                    st.caption(f"📄 {clean_source}")
                    st.caption("👆 클릭하여 크게 보기")
    
    return response_text

def main() -> None:
    """
    Streamlit 메인 애플리케이션 함수
    
    RAG 챗봇의 전체 사용자 인터페이스를 구성하고 실행합니다.
    사이드바의 설정 패널과 메인 채팅 인터페이스를 포함합니다.
    """
    st.title(settings.app_title)
    st.markdown(settings.app_description)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 문서 관리 섹션
        st.subheader("📄 문서 관리")
        
        # OCR 옵션
        use_ocr = st.checkbox(
            "PDF OCR 사용", 
            value=True,
            help="스캔된 PDF나 이미지 PDF에서 텍스트를 추출하려면 체크하세요. (Tesseract 필요)"
        )
        
        # 에이전트 모드 옵션
        use_agent_mode = st.checkbox(
            "🤖 에이전트 모드 (고품질 처리)", 
            value=False,
            help="로컬 LLM을 활용한 지능형 문서 전처리를 사용합니다.\n• 한국어 텍스트 분절 문제 해결\n• 고유명사 완성도 향상\n• 문맥 연결성 개선\n⚠️ 처리 시간이 더 오래 걸립니다."
        )
        
        # 2단계 품질 개선 옵션 (기본값: True)
        enable_postprocessing = st.checkbox(
            "✨ PDF 변환 시 텍스트 품질 자동 개선 (권장)", 
            value=True,
            help="PDF에서 추출한 텍스트의 품질을 자동으로 개선합니다.\n• 한국어 문장 연결 및 띄어쓰기 교정\n• 문맥 일관성 향상\n• 품질 점수 90점 이상 달성\n• 처리된 파일은 processed_docs 폴더에 저장됩니다.",
            disabled=False
        )
        
        # 지능형 이미지 추출 옵션 (새로 추가)
        use_intelligent_extraction = st.checkbox(
            "🧠 지능형 이미지 추출 (실험적)", 
            value=False,
            help="로컬 AI 모델을 사용한 지능형 이미지 추출:\n• 문서 주제와 관련된 이미지만 추출\n• 텍스트 이미지는 OCR로 자동 변환\n• Midm-2.0 한국어 모델 사용\n• Gemma-2 멀티모달 모델 사용\n⚠️ 모델 다운로드 필요 (약 12GB)"
        )
        
        # 현재 설정 상태 표시
        if use_agent_mode:
            st.info("🤖 **에이전트 모드 활성화**: 고품질 전처리 사용 중")
        if enable_postprocessing:
            st.info("✨ **2단계 품질 개선 활성화**: PDF 변환 후 자동으로 텍스트 품질을 개선합니다.")
        st.divider()
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "문서 업로드 (TXT, MD, PDF, DOCX) 🆕 이미지 추출 지원",
            type=['txt', 'md', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="PDF/DOCX 파일의 이미지도 자동 추출됩니다"
        )
        
        if uploaded_files:
            if st.button("문서 처리 및 저장"):
                # DocumentLoader를 현재 옵션으로 재초기화
                if (use_agent_mode != st.session_state.document_loader.use_agent_preprocessing or
                    enable_postprocessing != getattr(st.session_state.document_loader, 'enable_postprocessing', False) or
                    use_intelligent_extraction != getattr(st.session_state.document_loader, 'use_intelligent_image_extraction', False)):
                    st.session_state.document_loader = DocumentLoader(
                        use_ocr=st.session_state.document_loader.use_ocr,
                        use_agent_preprocessing=use_agent_mode,
                        enable_postprocessing=enable_postprocessing,
                        use_intelligent_image_extraction=use_intelligent_extraction
                    )
                
                # 디렉토리 준비
                DocumentProcessor.prepare_directories()
                
                # 전체 처리 통계 초기화
                total_files = len(uploaded_files)
                processed_files = 0
                failed_files = 0
                total_chunks = 0
                total_processing_time = 0
                
                # 전체 진행률 표시
                overall_progress = st.progress(0)
                overall_status = st.empty()
                
                for file_idx, uploaded_file in enumerate(uploaded_files, 1):
                    # 개별 파일 처리 시작
                    overall_status.text(f"파일 {file_idx}/{total_files} 처리 중: {uploaded_file.name}")
                    overall_progress.progress(file_idx / total_files)
                    
                    # 파일별 상세 정보 표시
                    file_container = st.container()
                    with file_container:
                        st.divider()
                        file_col1, file_col2 = st.columns([3, 1])
                        
                        with file_col1:
                            st.write(f"**📄 [{file_idx}/{total_files}] {uploaded_file.name}**")
                        
                        with file_col2:
                            file_size_mb = uploaded_file.size / (1024 * 1024)
                            st.caption(f"{file_size_mb:.1f}MB")
                        
                        # 개별 파일 진행 상황
                        file_progress = st.progress(0)
                        file_status = st.empty()
                        
                        # 임시 파일로 저장
                        temp_path = TEMP_DOCUMENT_PATH.format(filename=uploaded_file.name)
                        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                        
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # 진행 상황 콜백
                        def update_progress(progress, message):
                            file_progress.progress(progress)
                            file_status.text(message)
                        
                        # 문서 처리 시작
                        start_time = time.time()
                        processing_error = None
                        documents = None
                        processing_time = 0.0  # 초기값 설정
                        
                        try:
                            logger.info(f"문서 처리 시작: {uploaded_file.name} ({file_size_mb:.1f}MB)")
                            
                            # DocumentLoader 설정 업데이트
                            current_ocr = getattr(st.session_state.document_loader, 'use_ocr', True)
                            current_agent = getattr(st.session_state.document_loader, 'use_agent_preprocessing', False)
                            current_intelligent = getattr(st.session_state.document_loader, 'use_intelligent_image_extraction', False)
                            
                            if current_ocr != use_ocr or current_agent != use_agent_mode or current_intelligent != use_intelligent_extraction:
                                st.session_state.document_loader = DocumentLoader(
                                    use_ocr=use_ocr, 
                                    use_agent_preprocessing=use_agent_mode,
                                    use_intelligent_image_extraction=use_intelligent_extraction
                                )
                            
                            # 벡터 DB에 추가하기 전 청크 수 확인
                            before_count = safe_get_vector_db().get_document_count()
                            
                            # 문서 로드
                            documents = st.session_state.document_loader.load_document(temp_path, update_progress)
                            
                            # 지능형 이미지 추출 결과 확인
                            image_extraction_success = False
                            extracted_images_count = 0
                            if use_intelligent_extraction:
                                # 이미지 추출 결과 확인
                                from pathlib import Path
                                pdf_name = Path(uploaded_file.name).stem
                                images_dir = Path(f"data/extracted_images/{pdf_name}/images")
                                if images_dir.exists():
                                    extracted_images = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
                                    extracted_images_count = len(extracted_images)
                                    if extracted_images_count > 0:
                                        image_extraction_success = True
                            
                            # 문서가 있거나 이미지 추출이 성공한 경우
                            if documents or image_extraction_success:
                                # 문서가 있으면 벡터 DB에 추가
                                if documents:
                                    # 벡터 DB에 추가
                                    update_progress(0.95, "벡터 데이터베이스에 저장 중...")
                                    safe_get_vector_db().add_documents(documents)
                                    
                                    # 저장 후 청크 수 확인 (품질 검증)
                                    after_count = safe_get_vector_db().get_document_count()
                                    saved_chunks = after_count - before_count
                                else:
                                    # 이미지만 추출된 경우
                                    saved_chunks = 0
                                
                                # 처리 완료
                                processing_time = time.time() - start_time
                                update_progress(1.0, "완료!")
                                
                                # 통계 업데이트
                                processed_files += 1
                                if documents:
                                    total_chunks += len(documents)
                                total_processing_time += processing_time
                                
                                # 청크 분석
                                chunk_analysis = analyze_chunks(documents) if documents else None
                                pdf_metadata = get_pdf_metadata(documents) if documents else None
                                
                                # 성공 메시지와 상세 정보 표시
                                success_col1, success_col2 = st.columns([2, 1])
                                
                                with success_col1:
                                    st.success(f"✅ {uploaded_file.name} 처리 완료")
                                    
                                    # 지능형 이미지 추출 정보 표시
                                    if image_extraction_success and not documents:
                                        st.caption(f"**처리 정보:** 지능형 이미지 추출 | {extracted_images_count}개 이미지 추출")
                                    # 처리 정보 표시
                                    elif pdf_metadata:
                                        info_text = f"**처리 정보:** {pdf_metadata['extraction_method']}"
                                        if pdf_metadata['page_count']:
                                            info_text += f" | {pdf_metadata['page_count']}페이지"
                                        
                                        processing_method = pdf_metadata.get('processing_method', '')
                                        if processing_method == 'markdown_optimized':
                                            info_text += " | 마크다운 최적화"
                                        elif 'agent_based' in processing_method:
                                            info_text += " | 🤖 에이전트 기반 고품질 변환"
                                            if pdf_metadata.get('image_count', 0) > 0:
                                                info_text += f" | {pdf_metadata['image_count']}개 이미지 추출"
                                        elif processing_method == 'improved_pdf_converter_with_images':
                                            info_text += " | 개선된 PDF 변환 (문장 연결성 향상)"
                                            if pdf_metadata.get('image_count', 0) > 0:
                                                info_text += f" | {pdf_metadata['image_count']}개 이미지 추출"
                                        st.caption(info_text)
                                
                                with success_col2:
                                    st.metric("처리 시간", f"{processing_time:.1f}초")
                                
                                # 청크 분석 정보
                                if chunk_analysis:
                                    chunk_col1, chunk_col2, chunk_col3, chunk_col4 = st.columns(4)
                                    
                                    with chunk_col1:
                                        st.metric("청크 수", f"{chunk_analysis['count']}개")
                                    
                                    with chunk_col2:
                                        st.metric("평균 크기", f"{chunk_analysis['avg_size']:.0f}자")
                                    
                                    with chunk_col3:
                                        st.metric("최소 크기", f"{chunk_analysis['min_size']}자")
                                    
                                    with chunk_col4:
                                        st.metric("최대 크기", f"{chunk_analysis['max_size']}자")
                                
                                # 품질 검증 결과
                                if documents:
                                    if len(documents) == saved_chunks:
                                        st.info(f"💾 품질 검증: 모든 청크({len(documents)}개)가 성공적으로 저장됨")
                                    else:
                                        st.warning(f"⚠️ 품질 검증: 로드된 청크({len(documents)}개) vs 저장된 청크({saved_chunks}개)")
                                elif image_extraction_success:
                                    st.info(f"🖼️ 이미지 추출: {extracted_images_count}개 이미지가 성공적으로 추출됨")
                                
                                # 검색 품질 테스트
                                search_results = test_search_quality(safe_get_vector_db(), uploaded_file.name)
                                if search_results:
                                    with st.expander("🔍 검색 테스트 결과"):
                                        search_cols = st.columns(len(search_results))
                                        for i, (keyword, result_count) in enumerate(search_results.items()):
                                            with search_cols[i]:
                                                if isinstance(result_count, int):
                                                    st.metric(f"'{keyword}'", f"{result_count}개 결과")
                                                else:
                                                    st.caption(f"'{keyword}': {result_count}")
                                
                                if documents:
                                    logger.info(f"문서 처리 완료: {uploaded_file.name} - {len(documents)} 청크, 처리시간: {processing_time:.1f}초")
                                else:
                                    logger.info(f"이미지 추출 완료: {uploaded_file.name} - {extracted_images_count} 이미지, 처리시간: {processing_time:.1f}초")
                            
                            else:
                                failed_files += 1
                                processing_error = "문서를 로드할 수 없음"
                                processing_time = time.time() - start_time
                                
                        except Exception as e:
                            failed_files += 1
                            processing_error = str(e)
                            processing_time = time.time() - start_time
                            logger.error(f"문서 처리 실패: {uploaded_file.name} - {processing_error}")
                        
                        # 처리 실패 시 에러 표시
                        if processing_error:
                            st.error(f"❌ {uploaded_file.name} 처리 실패: {processing_error}")
                            st.caption(f"처리 시간: {processing_time:.1f}초")
                        
                        # 진행 바와 상태 텍스트 제거
                        file_progress.empty()
                        file_status.empty()
                        
                        # 임시 파일 정리
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                
                # 전체 처리 완료 후 통계 표시
                overall_progress.empty()
                overall_status.empty()
                
                st.divider()
                st.subheader("📊 전체 처리 결과")
                
                # 전체 통계
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("총 파일", f"{total_files}개")
                
                with stat_col2:
                    st.metric("처리 성공", f"{processed_files}개", 
                             delta=f"{failed_files}개 실패" if failed_files > 0 else "모두 성공")
                
                with stat_col3:
                    st.metric("총 청크", f"{total_chunks}개")
                
                with stat_col4:
                    st.metric("총 처리시간", f"{total_processing_time:.1f}초")
                
                # 최종 벡터 DB 상태
                final_doc_count = safe_get_vector_db().get_document_count()
                if total_chunks > 0:
                    st.success(f"🎉 업로드 완료! 벡터 데이터베이스에 총 {final_doc_count}개의 청크가 저장되어 있습니다.")
                else:
                    st.warning("⚠️ 처리된 문서가 없습니다.")
                
                # 처리 완료 후 새로고침
                time.sleep(2)  # 사용자가 결과를 확인할 수 있도록 대기
                st.rerun()
        
        # 기존 문서 로드
        if st.button("domain.md 파일 로드"):
            domain_path = "./domain.md"
            if os.path.exists(domain_path):
                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    documents = st.session_state.document_loader.load_document(domain_path, update_progress)
                    
                    update_progress(0.95, "벡터 데이터베이스에 저장 중...")
                    safe_get_vector_db().add_documents(documents)
                    
                    update_progress(1.0, "완료!")
                    st.success(f"✅ domain.md 로드 완료 ({len(documents)} 청크)")
                    
                    # 진행 표시 제거
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    st.error(f"❌ 파일 로드 실패: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.error("domain.md 파일을 찾을 수 없습니다.")
        
        # 벡터 DB 상태
        st.divider()
        vector_db = safe_get_vector_db()
        doc_count = vector_db.get_document_count()
        st.info(f"💾 저장된 문서 청크: {doc_count}개")
        
        # 로그 뷰어 (확장 가능)
        with st.expander("📋 처리 로그 보기"):
            log_file = LOG_FILE_PATTERN.format(date=datetime.now().strftime('%Y%m%d'))
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    # 최근 N줄만 표시
                    lines = f.readlines()
                    recent_lines = lines[-MAX_LOG_LINES_DISPLAY:] if len(lines) > MAX_LOG_LINES_DISPLAY else lines
                    st.text(''.join(recent_lines))
            else:
                st.text("로그 파일이 없습니다.")
        
        # 벡터 DB 초기화
        col1, col2 = st.columns(2)
        with col1:
            if st.button("벡터 DB 초기화", type="secondary", key="clear_db_btn"):
                st.session_state.show_clear_confirm = True
        
        if 'show_clear_confirm' in st.session_state and st.session_state.show_clear_confirm:
            with col2:
                if st.button("⚠️ 확인", type="primary", key="confirm_clear"):
                    safe_get_vector_db().clear_database()
                    # RAG 체인도 재초기화 (벡터 DB 인스턴스 공유)
                    st.session_state.rag_chain = RAGChain(
                        provider=st.session_state.current_provider, 
                        model=st.session_state.current_model,
                        vector_db=safe_get_vector_db()
                    )
                    st.success("벡터 데이터베이스가 초기화되었습니다.")
                    st.session_state.show_clear_confirm = False
                    st.rerun()
        
        # LLM 설정
        st.divider()
        st.subheader("🤖 LLM 모델 설정")
        
        # 사용 가능한 모델 가져오기
        available_models = safe_get_rag_chain().get_available_models()
        
        # LLM 제공자 선택
        provider_names = {
            "openai": "OpenAI",
            "google": "Google Gemini",
            "anthropic": "Anthropic Claude",
            "local": "로컬 LLM"
        }
        
        selected_provider = st.selectbox(
            "LLM 제공자 선택",
            options=list(provider_names.keys()),
            format_func=lambda x: provider_names[x],
            index=list(provider_names.keys()).index(st.session_state.current_provider)
        )
        
        # 선택된 제공자의 모델 목록
        if selected_provider in available_models:
            model_options = available_models[selected_provider]
            # 로컬 LLM: EXAONE(Transformers) + Local GGUF 모두 선택 가능
            if selected_provider == "local":
                def is_local_allowed(entry: dict) -> bool:
                    model_id = str(entry.get('model', '')).lower()
                    name = str(entry.get('name', '')).lower()
                    return (
                        "exaone" in model_id or "exaone" in name or
                        "gguf" in model_id or "gguf" in name or
                        model_id in ("local-gguf", "local-model")
                    )
                filtered_options = [m for m in model_options if is_local_allowed(m)]
                # 정렬: Local GGUF 우선 표시 → EXAONE 순
                def sort_key(e: dict) -> int:
                    txt = (str(e.get('name','')) + str(e.get('model',''))).lower()
                    if 'gguf' in txt or e.get('model') in ("local-gguf", "local-model"):
                        return 0
                    if 'exaone' in txt:
                        return 1
                    return 2
                filtered_options = sorted(filtered_options, key=sort_key)
                if not filtered_options:
                    filtered_options = [{
                        "name": "Local GGUF (llama.cpp)",
                        "model": "local-gguf",
                        "description": "local_models 내 GGUF 자동 탐색"
                    }]
                model_options = filtered_options
            
            # 모델 선택
            selected_model_info = st.selectbox(
                "모델 선택",
                options=model_options,
                format_func=lambda x: f"{x['name']} - {x['description']}",
                index=0
            )
            
            selected_model = selected_model_info['model'] if selected_model_info else None
            
            # API 키 확인
            api_key_status = "✅ API 키 설정됨"
            if selected_provider == "openai" and not settings.openai_api_key:
                api_key_status = "❌ OpenAI API 키가 필요합니다"
            elif selected_provider == "google" and not settings.google_api_key:
                api_key_status = "❌ Google API 키가 필요합니다"
            elif selected_provider == "anthropic" and not settings.anthropic_api_key:
                api_key_status = "❌ Anthropic API 키가 필요합니다"
            elif selected_provider == "local":
                api_key_status = "✅ 로컬 모델 (API 키 불필요)"
            
            st.caption(api_key_status)
            
            # 모델 변경 버튼
            if st.button("모델 적용", type="primary"):
                try:
                    st.session_state.rag_chain.update_llm(selected_provider, selected_model)
                    st.session_state.current_provider = selected_provider
                    st.session_state.current_model = selected_model
                    st.success(f"✅ {provider_names[selected_provider]} - {selected_model_info['name']} 모델로 변경되었습니다!")
                except Exception as e:
                    st.error(f"❌ 모델 변경 실패: {str(e)}")
        
        # 고급 설정
        st.divider()
        st.subheader("🔧 고급 설정")
        
        k_documents = st.slider(
            "검색할 문서 개수",
            min_value=1,
            max_value=10,
            value=settings.k_documents,
            help="더 많은 문서를 검색하면 더 포괄적인 답변을 얻을 수 있습니다."
        )
        settings.k_documents = k_documents
        
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.temperature,
            step=0.1,
            help="낮을수록 일관성 있고, 높을수록 창의적인 답변"
        )
        settings.temperature = temperature
        
        # 스트리밍 설정
        enable_streaming = st.checkbox(
            "🚀 실시간 답변 (스트리밍)", 
            value=settings.enable_streaming,
            help="답변이 실시간으로 타이핑되듯이 나타납니다. ChatGPT와 같은 경험을 제공합니다."
        )
        settings.enable_streaming = enable_streaming
        
        # 디버그 모드
        debug_mode = st.checkbox("🐛 디버그 모드", help="검색 결과와 점수를 표시합니다.")
        st.session_state.debug_mode = debug_mode
    
    # 메인 챗 인터페이스
    st.header("💬 챗봇")
    
    # 전역 CSS 및 JavaScript 추가 (이미지 라이트박스용)
    st.markdown("""
        <style>
        /* 이미지 컨테이너 스타일 - 가장 강력한 선택자 사용 */
        div[onclick*="openLightbox"],
        .stMarkdown div[onclick*="openLightbox"],
        [data-testid="stMarkdownContainer"] div[onclick*="openLightbox"],
        .image-container {
            width: 300px !important;
            height: 240px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border: 2px solid #ddd !important;
            border-radius: 8px !important;
            margin: 0 auto 8px auto !important;
            overflow: hidden !important;
            background-color: #f8f9fa !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-sizing: border-box !important;
        }
        .image-container:hover {
            border-color: #007acc !important;
            box-shadow: 0 4px 8px rgba(0,122,204,0.2) !important;
            transform: translateY(-2px) !important;
        }
        .image-container img,
        .image-thumbnail {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
            display: block !important;
            border-radius: 6px !important;
        }
        /* 라이트박스 스타일 */
        .lightbox {
            display: none;
            position: fixed;
            z-index: 999999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.9);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .lightbox.show {
            display: block;
            opacity: 1;
        }
        .lightbox-content {
            position: relative;
            margin: auto;
            padding: 0;
            width: 90%;
            max-width: 900px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .lightbox-image {
            width: auto;
            height: auto;
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }
        .lightbox-close {
            position: absolute;
            top: 20px;
            right: 35px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
            transition: color 0.3s;
        }
        .lightbox-close:hover {
            color: #ccc;
        }
        .lightbox-info {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            text-align: center;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 14px;
        }
        </style>
        
        <script>
        function openLightbox(imgSrc, filename, source) {
            var lightbox = document.createElement('div');
            lightbox.className = 'lightbox';
            lightbox.innerHTML = `
                <span class="lightbox-close">&times;</span>
                <div class="lightbox-content">
                    <img class="lightbox-image" src="${imgSrc}" alt="Image">
                    <div class="lightbox-info">
                        <div><strong>📄 ${source}</strong></div>
                        <div>🖼️ ${filename}</div>
                    </div>
                </div>
            `;
            document.body.appendChild(lightbox);
            
            setTimeout(function() {
                lightbox.classList.add('show');
            }, 10);
            
            function closeLightbox() {
                lightbox.classList.remove('show');
                setTimeout(function() {
                    if (lightbox && lightbox.parentNode) {
                        document.body.removeChild(lightbox);
                    }
                }, 300);
            }
            
            lightbox.querySelector('.lightbox-close').onclick = closeLightbox;
            lightbox.onclick = function(e) {
                if (e.target === lightbox) {
                    closeLightbox();
                }
            };
            
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    closeLightbox();
                }
            }, {once: true});
        }
        // 안전한 이벤트 바인딩: onClick 속성 대신 JS로 바인딩(React 오류 방지)
        (function bindLightbox(){
            function attach(el){
                if (!el || el.dataset.lbBound === '1') return;
                el.addEventListener('click', function(){
                    const src = el.getAttribute('data-img-src');
                    const fn = el.getAttribute('data-filename') || '';
                    const sc = el.getAttribute('data-source') || '';
                    if (src) { openLightbox(src, fn, sc); }
                });
                el.dataset.lbBound = '1';
            }
            document.querySelectorAll('.image-container').forEach(attach);
            const obs = new MutationObserver(function(muts){
                muts.forEach(function(m){
                    m.addedNodes && m.addedNodes.forEach(function(n){
                        if (n && n.nodeType === 1){
                            if (n.classList && n.classList.contains('image-container')) attach(n);
                            if (n.querySelectorAll) n.querySelectorAll('.image-container').forEach(attach);
                        }
                    });
                });
            });
            obs.observe(document.body, {childList:true, subtree:true});
        })();
        </script>
        """, unsafe_allow_html=True)

    # 현재 사용 중인 모델 및 컨텍스트 사용량 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        current_model_display = f"현재 모델: **{provider_names.get(st.session_state.current_provider, st.session_state.current_provider)}**"
        if st.session_state.current_model:
            current_model_display += f" - {st.session_state.current_model}"
        st.caption(current_model_display)
    
    with col2:
        # 컨텍스트 사용량 계산 및 표시
        if st.session_state.messages:
            # 현재 대화 내역에서 텍스트 추출
            messages_for_count = []
            for msg in st.session_state.messages:
                messages_for_count.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 마지막 검색된 문서의 토큰 수 (대략적)
            documents_text = " " * (st.session_state.rag_chain.get_last_context_tokens() * 4)
            
            # 컨텍스트 사용량 계산
            usage_info = st.session_state.token_counter.calculate_context_usage(
                messages_for_count,
                documents_text,
                st.session_state.current_provider,
                st.session_state.current_model or "gpt-3.5-turbo"
            )
            
            # 사용량 표시
            usage_display = st.session_state.token_counter.format_token_display(usage_info)
            st.caption(f"컨텍스트: {usage_display}")
            
            # 상세 정보 툴팁
            with st.expander("📊 토큰 사용량 상세"):
                st.markdown(st.session_state.token_counter.get_usage_breakdown(usage_info))
        else:
            st.caption("컨텍스트: 🟢 0 토큰 사용 중")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 어시스턴트 응답에 이미지 표시
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                # 관련 이미지 수집 및 표시
                related_images = []
                for source in message["sources"]:
                    # source가 dict인지 확인하고 images 필드 찾기
                    if isinstance(source, dict):
                        # 직접 images 필드가 있는 경우
                        images = source.get('images', [])
                        source_name = source.get('file_name', 'Unknown')
                    else:
                        # Document 객체인 경우 metadata에서 찾기
                        metadata = getattr(source, 'metadata', {})
                        images = metadata.get('images', [])
                        source_name = metadata.get('file_name', 'Unknown')
                    
                    for image_info in images:
                        image_path = image_info.get('path', '')
                        if os.path.exists(image_path):
                            related_images.append({
                                'path': image_path,
                                'filename': image_info.get('filename', 'Unknown'),
                                'source': source_name,
                                'description': image_info.get('description')
                            })
                
                # 이미지 표시
                if related_images:
                    # display_images_in_response 함수는 2개 파라미터를 받으므로 기존 방식 유지
                    st.markdown("### 📸 관련 이미지")
                    
                    # 2-column 레이아웃으로 이미지 표시 (최대 6개까지)
                    images_to_show = related_images[:6]
                    cols = st.columns(2)
                    
                    for i, image_info in enumerate(images_to_show):
                        col_idx = i % 2
                        
                        with cols[col_idx]:
                            image_data_url = serve_image(image_info['path'])
                            if image_data_url:
                                # 고정 크기 컨테이너로 이미지 표시
                                clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                                clean_source = image_info.get('source', 'Unknown').replace('+', ' ')
                                if len(clean_source) > 35:
                                    clean_source = clean_source[:32] + "..."
                                
                                image_id = f"image_{hash(image_info['path'])}"
                                
                                # CSS 스타일과 JavaScript는 이미 정의되어 있음
                                st.markdown(
                                    f'''<div class="image-container" data-img-src="{image_data_url}" data-filename="{clean_filename}" data-source="{clean_source}" title="클릭하여 크게 보기">
                                        <img id="{image_id}" 
                                            src="{image_data_url}" 
                                            class="image-thumbnail">
                                    </div>''',
                                    unsafe_allow_html=True
                                )
                                # 설명 캡션
                                if image_info.get('description'):
                                    st.caption(image_info.get('description'))
                                st.caption(f"📄 {clean_source}")
                                st.caption("👆 클릭하여 크게 보기")
            
            # 출처 정보 표시
            if "sources" in message and message["sources"]:
                with st.expander("📌 참고 문서"):
                    for i, source in enumerate(message["sources"]):
                        if isinstance(source, dict):
                            file_name = source.get('file_name', 'Unknown')
                            relevance_score = source.get('relevance_score', 0)
                            content_preview = source.get('content_preview', '')
                        else:
                            metadata = getattr(source, 'metadata', {})
                            file_name = metadata.get('file_name', 'Unknown')
                            relevance_score = 0
                            content_preview = source.page_content[:200] if hasattr(source, 'page_content') else ''
                        
                        st.markdown(f"**문서 {i+1}**: {file_name}")
                        if relevance_score > 0:
                            st.markdown(f"- 관련도: {1 - relevance_score:.2%}")
                        st.markdown(f"- 내용 미리보기: {content_preview}")
                        st.divider()
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 봇 응답 생성 (스트리밍/일반 모드 선택)
        with st.chat_message("assistant"):
            # 공통 변수 초기화
            context_tokens = 0
            response = {"status": "error", "answer": "", "sources": []}
            
            if settings.enable_streaming:
                # 🚀 스트리밍 모드
                status_placeholder = st.empty()
                response_placeholder = st.empty()
                
                # 로컬 모델 사용 시 추가 대기 메시지
                if st.session_state.current_provider == "local":
                    status_placeholder.warning("⏳ 로컬 모델을 사용 중입니다. 응답 시간이 다소 걸릴 수 있습니다. 잠시만 기다려주세요...")
                
                # 스트리밍 응답 변수들
                full_response = ""
                response_sources = []
                response_status = "unknown"
                response_search_info = {}
                
                # 스트리밍 쿼리 실행
                try:
                    def stream_generator():
                        """스트리밍 제너레이터"""
                        for chunk in st.session_state.rag_chain.stream_query(prompt):
                            yield chunk
                    
                    # 스트리밍 응답 처리
                    for chunk in stream_generator():
                        chunk_type = chunk.get("type", "unknown")
                        chunk_content = chunk.get("content", "")
                        
                        if chunk_type == "status":
                            # 상태 메시지 표시
                            if st.session_state.current_provider == "local":
                                # 로컬 모델용 상태 메시지 커스터마이징
                                if "검색" in chunk_content:
                                    status_placeholder.info(chunk_content)
                                elif "생성" in chunk_content:
                                    status_placeholder.warning("⏳ 로컬 모델이 답변을 생성 중입니다. 시간이 걸릴 수 있습니다...")
                            else:
                                status_placeholder.info(chunk_content)
                            
                        elif chunk_type == "content":
                            # 실시간 답변 내용 추가
                            full_response = chunk.get("full_content", full_response + chunk_content)
                            response_placeholder.markdown(full_response + "▌")  # 커서 효과
                            
                        elif chunk_type == "complete":
                            # 스트리밍 완료 - 최종 정보 수집
                            full_response = chunk.get("full_content", full_response)
                            response_sources = chunk.get("sources", [])
                            response_status = chunk.get("status", "success")
                            response_search_info = chunk.get("search_info", {})
                            context_tokens = chunk.get("context_tokens", 0)
                            
                            # 상태 메시지 제거하고 최종 답변 표시
                            status_placeholder.empty()
                            response_placeholder.markdown(full_response)
                            
                        elif chunk_type == "error":
                            # 에러 처리
                            full_response = chunk_content
                            response_sources = chunk.get("sources", [])
                            response_status = chunk.get("status", "error")
                            
                            # 상태 메시지 제거하고 에러 메시지 표시
                            status_placeholder.empty()
                            response_placeholder.error(full_response)
                            
                            break
                    
                    # 응답 구조 생성 (기존 코드와 호환성 유지)
                    response = {
                        "answer": full_response,
                        "sources": response_sources,
                        "status": response_status,
                        "search_info": response_search_info
                    }
                    
                    # 스트리밍 완료 후 관련 이미지 표시
                    if response_status == "success" and response_sources:
                        # 이미지 정보 수집
                        related_images = []
                        for source in response_sources:
                            # source가 dict인지 확인하고 images 필드 찾기
                            if isinstance(source, dict):
                                images = source.get('images', [])
                                source_name = source.get('file_name', 'Unknown')
                            else:
                                # Document 객체인 경우 metadata에서 찾기
                                metadata = getattr(source, 'metadata', {})
                                images = metadata.get('images', [])
                                source_name = metadata.get('file_name', 'Unknown')
                            
                            for image_info in images:
                                image_path = image_info.get('path', '')
                                if os.path.exists(image_path):
                                    related_images.append({
                                        'path': image_path,
                                        'filename': image_info.get('filename', 'Unknown'),
                                        'source': source_name,
                                        'description': image_info.get('description')
                                    })
                        
                        # 이미지 표시
                        if related_images:
                            st.markdown("### 📸 관련 이미지")
                            
                            # 2-column 레이아웃으로 이미지 표시 (최대 6개까지)
                            images_to_show = related_images[:6]
                            cols = st.columns(2)
                            
                            for i, image_info in enumerate(images_to_show):
                                col_idx = i % 2
                                
                                with cols[col_idx]:
                                    image_data_url = serve_image(image_info['path'])
                                    if image_data_url:
                                        # 고정 크기 컨테이너로 이미지 표시
                                        clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                                        clean_source = image_info.get('source', 'Unknown').replace('+', ' ')
                                        if len(clean_source) > 35:
                                            clean_source = clean_source[:32] + "..."
                                        
                                        image_id = f"image_{hash(image_info['path'])}"
                                        
                                        # CSS 스타일과 JavaScript는 이미 정의되어 있음
                                        st.markdown(
                                            f'''<div class="image-container" data-img-src="{image_data_url}" data-filename="{clean_filename}" data-source="{clean_source}" title="클릭하여 크게 보기">
                                                <img id="{image_id}" 
                                                    src="{image_data_url}" 
                                                    class="image-thumbnail">
                                            </div>''',
                                            unsafe_allow_html=True
                                        )
                                        if image_info.get('description'):
                                            st.caption(image_info.get('description'))
                                        st.caption(f"📄 {clean_source}")
                                        st.caption("👆 클릭하여 크게 보기")
                    
                except Exception as e:
                    # 스트리밍 에러 처리
                    logger.error(f"스트리밍 중 오류 발생: {str(e)}")
                    status_placeholder.empty()
                    response_placeholder.error(f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}")
                    
                    response = {
                        "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                        "sources": [],
                        "status": "error"
                    }
                    
            else:
                # 📝 일반 모드 (기존 방식)
                # 로컬 모델 사용 시 특별 메시지
                if st.session_state.current_provider == "local":
                    spinner_text = "⏳ 로컬 모델이 답변을 생성하는 중... (최대 60초까지 걸릴 수 있습니다)"
                else:
                    spinner_text = "답변을 생성하는 중..."
                
                with st.spinner(spinner_text):
                    response = st.session_state.rag_chain.query(prompt)
                
                # 실제 검색된 문서의 토큰 수 가져오기 (안전하게)
                try:
                    context_tokens = st.session_state.rag_chain.get_last_context_tokens()
                except AttributeError:
                    context_tokens = 0
                
                # 답변 표시
                if response["status"] == "success":
                    st.markdown(response["answer"])
                    
                    # 관련 이미지 표시
                    if 'sources' in response and response['sources']:
                        # 이미지 정보 수집
                        related_images = []
                        for source in response['sources']:
                            # source가 dict인지 확인하고 images 필드 찾기
                            if isinstance(source, dict):
                                images = source.get('images', [])
                                source_name = source.get('file_name', 'Unknown')
                            else:
                                # Document 객체인 경우 metadata에서 찾기
                                metadata = getattr(source, 'metadata', {})
                                images = metadata.get('images', [])
                                source_name = metadata.get('file_name', 'Unknown')
                            
                            for image_info in images:
                                image_path = image_info.get('path', '')
                                if os.path.exists(image_path):
                                    related_images.append({
                                        'path': image_path,
                                        'filename': image_info.get('filename', 'Unknown'),
                                        'source': source_name,
                                        'description': image_info.get('description')
                                    })
                        
                        # 이미지 표시
                        if related_images:
                            st.markdown("### 📸 관련 이미지")
                            
                            # 2-column 레이아웃으로 이미지 표시 (최대 6개까지)
                            images_to_show = related_images[:6]
                            cols = st.columns(2)
                            
                            for i, image_info in enumerate(images_to_show):
                                col_idx = i % 2
                                
                                with cols[col_idx]:
                                    image_data_url = serve_image(image_info['path'])
                                    if image_data_url:
                                        # 고정 크기 컨테이너로 이미지 표시
                                        clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                                        clean_source = image_info.get('source', 'Unknown').replace('+', ' ')
                                        if len(clean_source) > 35:
                                            clean_source = clean_source[:32] + "..."
                                        
                                        image_id = f"image_{hash(image_info['path'])}"
                                        
                                        # CSS 스타일과 JavaScript는 이미 정의되어 있음
                                    st.markdown(
                                        f'''<div class="image-container" data-img-src="{image_data_url}" data-filename="{clean_filename}" data-source="{clean_source}" title="클릭하여 크게 보기">
                                            <img id="{image_id}" 
                                                src="{image_data_url}" 
                                                class="image-thumbnail">
                                        </div>''',
                                        unsafe_allow_html=True
                                    )
                                    if image_info.get('description'):
                                        st.caption(image_info.get('description'))
                                    st.caption(f"📄 {clean_source}")
                                    st.caption("👆 클릭하여 크게 보기")
                else:
                    st.error(response["answer"])
                
            # 디버그 모드일 때 검색 결과 표시
            if st.session_state.debug_mode:
                    with st.expander("🔍 디버그 정보"):
                        st.write(f"**검색 쿼리**: {prompt}")
                        st.write(f"**벡터 DB 상태**:")
                        st.write(f"- 총 문서 청크 수: {safe_get_vector_db().get_document_count()}")
                        st.write(f"- 벡터 DB 타입: {settings.vector_db_type}")
                        st.write(f"- 검색 임계값: FAISS < 2.0, ChromaDB > 0.2")
                        st.write(f"**검색 설정**:")
                        st.write(f"- 검색할 문서 수 (k): {settings.k_documents}")
                        st.write(f"**검색 결과**:")
                        st.write(f"- 검색된 문서 수: {len(response.get('sources', []))}")
                        st.write(f"- 응답 상태: {response.get('status', 'unknown')}")
                        
                        if response.get('sources'):
                            st.write("**문서별 상세 정보**:")
                            for i, source in enumerate(response['sources']):
                                st.write(f"- **문서 {i+1}**: {source['file_name']}")
                                st.write(f"  - 관련도 점수: {source['relevance_score']:.4f}")
                                st.write(f"  - 소스 경로: {source['source_path']}")
                                st.write(f"  - 내용 미리보기: {source['content_preview'][:100]}...")
                                st.write("---")
                        else:
                            st.write("❌ 검색된 문서가 없습니다.")
                            
                            # 벡터 DB에 저장된 문서가 있다면 원시 검색 시도
                            if safe_get_vector_db().get_document_count() > 0:
                                try:
                                    vector_db = safe_get_vector_db()
                                    if hasattr(vector_db.vector_store, 'similarity_search_with_score'):
                                        raw_results = vector_db.vector_store.similarity_search_with_score(prompt, k=3)
                                        st.write(f"**원시 검색 결과** (임계값 무시):")
                                        for i, (doc, score) in enumerate(raw_results):
                                            st.write(f"- 문서 {i+1}: 점수 {score:.4f}, 내용: {doc.page_content[:100]}...")
                                except Exception as e:
                                    st.write(f"원시 검색 실패: {str(e)}")
            
            # 응답 처리 및 저장 (스트리밍 완료 후)
            if response["status"] == "success":
                # 응답 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
                # 관련 이미지 표시 (새로운 응답)
                if response["sources"]:
                    related_images = []
                    for source in response["sources"]:
                        # source가 dict인지 확인하고 images 필드 찾기
                        if isinstance(source, dict):
                            images = source.get('images', [])
                            source_name = source.get('file_name', 'Unknown')
                        else:
                            # Document 객체인 경우 metadata에서 찾기
                            metadata = getattr(source, 'metadata', {})
                            images = metadata.get('images', [])
                            source_name = metadata.get('file_name', 'Unknown')
                        
                        for image_info in images:
                            image_path = image_info.get('path', '')
                            if os.path.exists(image_path):
                                related_images.append({
                                    'path': image_path,
                                    'filename': image_info.get('filename', 'Unknown'),
                                    'source': source_name
                                })
                    
                    # 이미지 표시
                    if related_images:
                        # display_images_in_response 함수는 2개 파라미터를 받으므로 기존 방식 유지
                        st.markdown("### 📸 관련 이미지")
                        
                        # 2-column 레이아웃으로 이미지 표시 (최대 6개까지)
                        images_to_show = related_images[:6]
                        cols = st.columns(2)
                        
                        for i, image_info in enumerate(images_to_show):
                            col_idx = i % 2
                            
                            with cols[col_idx]:
                                image_data_url = serve_image(image_info['path'])
                                if image_data_url:
                                    # 고정 크기 컨테이너로 이미지 표시
                                    clean_filename = image_info.get('filename', 'Unknown').replace('+', ' ')
                                    clean_source = image_info.get('source', 'Unknown').replace('+', ' ')
                                    if len(clean_source) > 35:
                                        clean_source = clean_source[:32] + "..."
                                    
                                    image_id = f"image_{hash(image_info['path'])}"
                                    
                                    # CSS 스타일과 JavaScript는 이미 정의되어 있음
                                    st.markdown(
                                        f'''<div class="image-container" data-img-src="{image_data_url}" data-filename="{clean_filename}" data-source="{clean_source}" title="클릭하여 크게 보기">
                                            <img id="{image_id}" 
                                                src="{image_data_url}" 
                                                class="image-thumbnail">
                                        </div>''',
                                        unsafe_allow_html=True
                                    )
                                    st.caption(f"📄 {clean_source}")
                                    st.caption("👆 클릭하여 크게 보기")
                
                # 출처 정보 표시
                if response["sources"]:
                    with st.expander("📌 참고 문서"):
                        for i, source in enumerate(response["sources"]):
                            if isinstance(source, dict):
                                file_name = source.get('file_name', 'Unknown')
                                relevance_score = source.get('relevance_score', 0)
                                content_preview = source.get('content_preview', '')
                                relevance_percent = source.get('relevance_percent', None)
                            else:
                                metadata = getattr(source, 'metadata', {})
                                file_name = metadata.get('file_name', 'Unknown')
                                relevance_score = 0
                                content_preview = source.page_content[:200] if hasattr(source, 'page_content') else ''
                                relevance_percent = None
                            
                            st.markdown(f"**문서 {i+1}**: {file_name}")
                            if relevance_percent is not None:
                                st.markdown(f"- 관련도: {relevance_percent:.1f}%")
                            elif relevance_score > 0:
                                st.markdown(f"- 관련도: {(1 - relevance_score):.2%}")
                            st.markdown(f"- 내용 미리보기: {content_preview}")
                            st.divider()
            else:
                # 에러 상황일 때도 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })
            
            # 컨텍스트 사용량 업데이트를 위해 리런
            st.rerun()
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
