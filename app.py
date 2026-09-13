"""
RAG 챗봇 메인 애플리케이션 (리팩토링된 버전)

Streamlit 기반의 RAG(Retrieval-Augmented Generation) 챗봇 웹 애플리케이션입니다.
문서 업로드, 벡터 데이터베이스 관리, 다양한 LLM 모델 지원 등의 기능을 제공합니다.
"""

import logging
import os
import sys
from pathlib import Path

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

    if hasattr(torch._C, "_jit_set_emit_warnings"):
        torch._C._jit_set_emit_warnings(False)
    if hasattr(torch._C, "_set_print_stacktraces_on_fatal_signal"):
        torch._C._set_print_stacktraces_on_fatal_signal(False)
    if hasattr(torch, "_C") and hasattr(torch._C, "_set_print_warn"):
        try:
            torch._C._set_print_warn(False)
        except Exception:
            pass
except Exception:
    pass

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

# 프로젝트 임포트
from config import settings
from src.utils.logging_config import setup_logging, get_logger
from src.utils.tracing import ensure_langsmith_env

# UI 컴포넌트 임포트
from ui.components.sidebar import render_sidebar
from ui.components.chat_interface import render_chat_interface, render_chat_controls, render_feedback_interface
from ui.styles import inject_custom_css

# 파일 업로드 인터페이스는 사이드바로 이동됨
from ui.controllers.main_controller import MainController

# 로깅 설정
setup_logging(logging.DEBUG)  # DEBUG 레벨로 변경하여 상세 로그 출력
logger = get_logger(__name__)

# LangSmith 트레이싱 환경변수 반영 (config 설정을 실제 env로)
ensure_langsmith_env()

# 페이지 설정
st.set_page_config(
    page_title=settings.app_title,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Report a bug": None, "Get Help": None, "About": settings.app_description},
)


def initialize_application():
    """애플리케이션 초기화"""

    # 컨트롤러 초기화
    if "main_controller" not in st.session_state:
        st.session_state.main_controller = MainController()

    # 세션 상태 초기화
    st.session_state.main_controller.initialize_session_state()

    logger.info("애플리케이션 초기화 완료")


def render_header():
    """헤더 렌더링"""

    st.title("문서와 대화")

    controller = st.session_state.main_controller
    vector_db_status = controller.get_vector_db_status()
    model_info = controller.get_model_info()
    validation = controller.validate_configuration()

    db_text = f"{vector_db_status['document_count']}개 문서" if vector_db_status["is_initialized"] else "미초기화"
    model_text = (
        f"{model_info['provider']}: {model_info['model']}"
        if (model_info["provider"] and model_info["model"])
        else "미설정"
    )
    status_text = "시스템 정상" if validation["is_valid"] else f"문제 {len(validation['issues'])}개"

    st.caption(f"검색 가능: {db_text} · {status_text}")
    with st.expander("사용 중인 모델"):
        st.caption(model_text)


def render_main_tabs():
    """메인 탭 인터페이스 렌더링"""

    page = st.radio("화면 선택", ["채팅", "문서 관리"], horizontal=True, label_visibility="collapsed")
    controller = st.session_state.main_controller
    if page == "채팅":
        render_chat_tab(controller)
    else:
        render_document_tab(controller)


def render_chat_tab(controller: MainController):
    """채팅 탭 렌더링"""

    # 채팅 컨트롤
    render_chat_controls()

    # 채팅 인터페이스
    render_chat_interface(
        rag_chain=controller.safe_get_rag_chain(), sidebar_config=st.session_state.get("sidebar_config", {})
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
        if vector_db_status["is_initialized"]:
            st.success("✅ 데이터베이스 초기화됨")
            st.info(f"📄 총 문서 수: {vector_db_status['document_count']}개")
        else:
            st.warning("⚠️ 데이터베이스가 초기화되지 않았습니다")
            st.info("사이드바에서 문서를 업로드하여 시작하세요")

    with col2:
        if vector_db_status["is_initialized"]:
            # 데이터베이스 초기화 버튼
            if st.button("🗑️ 데이터베이스 초기화", type="secondary"):
                if st.session_state.get("confirm_reset", False):
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

    if st.session_state.get("debug_mode", False):
        with st.expander("🔧 디버그 정보", expanded=False):
            controller = st.session_state.main_controller

            # 시스템 상태
            st.subheader("시스템 상태")
            validation = controller.validate_configuration()

            if validation["issues"]:
                st.error("문제점:")
                for issue in validation["issues"]:
                    st.text(f"- {issue}")

            if validation["warnings"]:
                st.warning("경고:")
                for warning in validation["warnings"]:
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
    sidebar_config = st.session_state.get("sidebar_config", {})

    # 모델 변경 처리
    if "selected_provider" in sidebar_config and "selected_model" in sidebar_config:
        provider = sidebar_config["selected_provider"]
        model = sidebar_config["selected_model"]

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
            safe_get_vector_db=controller.safe_get_vector_db, safe_get_rag_chain=controller.safe_get_rag_chain
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

        inject_custom_css()

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
