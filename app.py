"""
RAG 챗봇 메인 애플리케이션 (리팩토링된 버전)

Streamlit 기반의 RAG(Retrieval-Augmented Generation) 챗봇 웹 애플리케이션입니다.
문서 업로드, 벡터 데이터베이스 관리, 다양한 LLM 모델 지원 등의 기능을 제공합니다.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# ChromaDB 텔레메트리 비활성화 (가장 먼저 실행)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# PyTorch torch.classes 경고 메시지 전역 억제
os.environ["TORCH_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_KERNEL_WARN"] = "0"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

import warnings

# 전체 PyTorch 및 transformers 경고 억제
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*torch.ops.*")
warnings.filterwarnings("ignore", message=".*torch.jit.*")
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*classes.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

# PyTorch 관련 로거 억제
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.jit").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Streamlit에서 torch.classes 경고를 원천 차단
try:
    import torch
    import torch._C
    if hasattr(torch._C, '_jit_set_emit_warnings'):
        torch._C._jit_set_emit_warnings(False)
    if hasattr(torch._C, '_set_print_stacktraces_on_fatal_signal'):
        torch._C._set_print_stacktraces_on_fatal_signal(False)
    if hasattr(torch, '_C') and hasattr(torch._C, '_set_print_warn'):
        try:
            torch._C._set_print_warn(False)
        except:
            pass
except Exception:
    pass

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 프로젝트 임포트
from config import settings
from src.utils.logging_config import setup_logging, get_logger

# UI 컴포넌트 임포트
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface, render_chat_controls, render_feedback_interface
# 파일 업로드 인터페이스는 사이드바로 이동됨
from ui.controllers.main_controller import MainController

# 로깅 설정
setup_logging(logging.INFO)
logger = get_logger(__name__)

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


def initialize_application():
    """애플리케이션 초기화"""
    
    # 컨트롤러 초기화
    if 'main_controller' not in st.session_state:
        st.session_state.main_controller = MainController()
    
    # 세션 상태 초기화
    st.session_state.main_controller.initialize_session_state()
    
    logger.info("애플리케이션 초기화 완료")


def render_header():
    """헤더 렌더링"""
    
    st.title("📚 RAG 챗봇")
    st.markdown("---")
    
    # 간단한 상태 정보 표시
    controller = st.session_state.main_controller
    vector_db_status = controller.get_vector_db_status()
    model_info = controller.get_model_info()
    
    # 상태 컬럼
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if vector_db_status['is_initialized']:
            st.success(f"🗃️ DB: {vector_db_status['document_count']}개 문서")
        else:
            st.warning("🗃️ DB: 미초기화")
    
    with col2:
        if model_info['provider'] and model_info['model']:
            status_icon = "🖥️" if model_info['is_local'] else "☁️"
            st.info(f"{status_icon} {model_info['provider']}: {model_info['model']}")
        else:
            st.warning("🤖 모델: 미설정")
    
    with col3:
        # 설정 검증 결과
        validation = controller.validate_configuration()
        if validation['is_valid']:
            st.success("✅ 시스템 정상")
        else:
            st.error(f"❌ 문제 {len(validation['issues'])}개")


def render_main_tabs():
    """메인 탭 인터페이스 렌더링"""
    
    # 탭 생성
    tab1, tab2 = st.tabs(["💬 채팅", "📁 문서 관리"])
    
    controller = st.session_state.main_controller
    
    with tab1:
        # 채팅 인터페이스
        render_chat_tab(controller)
    
    with tab2:
        # 파일 업로드 인터페이스
        render_document_tab(controller)


def render_chat_tab(controller: MainController):
    """채팅 탭 렌더링"""
    
    # 채팅 컨트롤
    render_chat_controls()
    
    # 채팅 인터페이스
    render_chat_interface(
        rag_chain=controller.safe_get_rag_chain(),
        sidebar_config=st.session_state.get('sidebar_config', {})
    )
    
    # 피드백 인터페이스
    render_feedback_interface()


def render_document_tab(controller: MainController):
    """문서 관리 탭 렌더링"""
    
    # 벡터 DB 상태 표시
    vector_db_status = controller.get_vector_db_status()
    
    st.subheader("📁 문서 데이터베이스 관리")
    
    # 문서 상태 정보
    col1, col2 = st.columns(2)
    
    with col1:
        if vector_db_status['is_initialized']:
            st.success(f"✅ 데이터베이스 초기화됨")
            st.info(f"📄 총 문서 수: {vector_db_status['document_count']}개")
        else:
            st.warning("⚠️ 데이터베이스가 초기화되지 않았습니다")
            st.info("사이드바에서 문서를 업로드하여 시작하세요")
    
    with col2:
        if vector_db_status['is_initialized']:
            # 데이터베이스 초기화 버튼
            if st.button("🗑️ 데이터베이스 초기화", type="secondary"):
                if st.session_state.get('confirm_reset', False):
                    vector_db = controller.safe_get_vector_db()
                    if vector_db:
                        vector_db.reset()
                        st.success("데이터베이스가 초기화되었습니다")
                        st.rerun()
                    st.session_state.confirm_reset = False
                else:
                    st.session_state.confirm_reset = True
                    st.warning("다시 클릭하면 모든 문서가 삭제됩니다")
    
    # 파일 업로드는 사이드바로 이동되었음을 안내
    st.markdown("---")
    st.info("💡 **파일 업로드**: 왼쪽 사이드바의 '문서 업로드' 섹션을 이용하세요")
    st.markdown("- PDF, DOCX, TXT, MD 파일을 지원합니다")
    st.markdown("- 이미지 추출 및 OCR 기능을 포함합니다")
    st.markdown("- 멀티모달 전처리를 지원합니다")


def render_debug_info():
    """디버그 정보 렌더링 (디버그 모드일 때만)"""
    
    if st.session_state.get('debug_mode', False):
        with st.expander("🔧 디버그 정보", expanded=False):
            controller = st.session_state.main_controller
            
            # 시스템 상태
            st.subheader("시스템 상태")
            validation = controller.validate_configuration()
            
            if validation['issues']:
                st.error("문제점:")
                for issue in validation['issues']:
                    st.text(f"- {issue}")
            
            if validation['warnings']:
                st.warning("경고:")
                for warning in validation['warnings']:
                    st.text(f"- {warning}")
            
            # 모델 정보
            st.subheader("모델 정보")
            model_info = controller.get_model_info()
            st.json(model_info)
            
            # 벡터 DB 상태
            st.subheader("벡터 DB 상태")
            vector_db_status = controller.get_vector_db_status()
            st.json(vector_db_status)
            
            # 세션 상태 키
            st.subheader("세션 상태 키")
            st.text(f"총 {len(st.session_state)} 개의 키:")
            for key in sorted(st.session_state.keys()):
                st.text(f"- {key}")


def handle_sidebar_changes():
    """사이드바 변경사항 처리"""
    
    controller = st.session_state.main_controller
    sidebar_config = st.session_state.get('sidebar_config', {})
    
    # 모델 변경 처리
    if 'selected_provider' in sidebar_config and 'selected_model' in sidebar_config:
        provider = sidebar_config['selected_provider']
        model = sidebar_config['selected_model']
        
        # 모델 변경이 있었는지 확인하고 처리
        success = controller.handle_provider_model_change(provider, model)
        if not success:
            st.error("모델 변경 실패")
    
    # 전처리 설정 변경 처리
    preprocessing_settings = controller.get_preprocessing_settings_from_sidebar(sidebar_config)
    controller.update_preprocessing_settings(preprocessing_settings)


def main():
    """메인 함수"""
    
    try:
        # 애플리케이션 초기화
        initialize_application()
        
        # 사이드바 렌더링
        controller = st.session_state.main_controller
        sidebar_config = render_sidebar(
            safe_get_vector_db=controller.safe_get_vector_db,
            safe_get_rag_chain=controller.safe_get_rag_chain,
            bootstrap_models_with_file_lock=controller.bootstrap_models_with_file_lock
        )
        
        # 사이드바 설정을 세션 상태에 저장
        st.session_state.sidebar_config = sidebar_config
        
        # 사이드바 변경사항 처리
        handle_sidebar_changes()
        
        # 헤더 렌더링
        render_header()
        
        # 메인 컨텐츠 렌더링
        render_main_tabs()
        
        # 디버그 정보 (필요시)
        render_debug_info()
        
        # CSS 스타일 주입 (이미지 라이트박스용)
        st.markdown("""
        <style>
        .image-container {
            position: relative;
            display: inline-block;
            margin: 5px;
            cursor: pointer;
        }
        
        .image-thumbnail {
            width: 200px;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        
        .image-thumbnail:hover {
            transform: scale(1.05);
        }
        
        .lightbox {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        
        .lightbox-content {
            margin: auto;
            display: block;
            width: 80%;
            max-width: 700px;
            max-height: 80%;
            object-fit: contain;
        }
        
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #bbb;
        }
        </style>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 이미지 클릭 이벤트 처리
            document.querySelectorAll('.image-container').forEach(function(container) {
                container.addEventListener('click', function() {
                    const imgSrc = this.getAttribute('data-img-src');
                    const filename = this.getAttribute('data-filename');
                    const source = this.getAttribute('data-source');
                    
                    // 라이트박스 생성
                    const lightbox = document.createElement('div');
                    lightbox.className = 'lightbox';
                    lightbox.style.display = 'block';
                    
                    lightbox.innerHTML = `
                        <span class="close">&times;</span>
                        <img class="lightbox-content" src="${imgSrc}" alt="${filename}">
                        <div style="text-align: center; color: white; margin-top: 10px;">
                            <p><strong>${filename}</strong></p>
                            <p>출처: ${source}</p>
                        </div>
                    `;
                    
                    document.body.appendChild(lightbox);
                    
                    // 닫기 이벤트
                    lightbox.querySelector('.close').addEventListener('click', function() {
                        document.body.removeChild(lightbox);
                    });
                    
                    lightbox.addEventListener('click', function(e) {
                        if (e.target === lightbox) {
                            document.body.removeChild(lightbox);
                        }
                    });
                });
            });
        });
        </script>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"애플리케이션 오류: {str(e)}")
        st.error(f"애플리케이션 오류가 발생했습니다: {str(e)}")
        
        # 디버그 정보 표시
        if st.checkbox("디버그 정보 표시"):
            import traceback
            st.text("상세 오류 정보:")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()