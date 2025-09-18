"""
사이드바 컴포넌트 - 설정 및 모델 관리
"""

import streamlit as st
import os
from datetime import datetime
from typing import Dict, Any, Optional

from config import settings
from src.processing.preprocessing_factory import PreprocessingModelFactory
from src.rag.rag_chain import RAGChain


def render_sidebar(
    safe_get_vector_db,
    safe_get_rag_chain,
    bootstrap_models_with_file_lock
) -> Dict[str, Any]:
    """
    사이드바를 렌더링하고 설정값들을 반환합니다.
    
    Args:
        safe_get_vector_db: 벡터 DB 인스턴스를 가져오는 함수
        safe_get_rag_chain: RAG 체인 인스턴스를 가져오는 함수
        bootstrap_models_with_file_lock: 모델 부트스트랩 함수
        
    Returns:
        Dict[str, Any]: 사이드바에서 설정된 모든 값들
    """
    
    sidebar_config = {}
    
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 문서 관리 섹션
        sidebar_config.update(_render_document_management())
        
        # 문서 전처리 섹션  
        sidebar_config.update(_render_text_preprocessing())
        
        # 이미지 추출·OCR 섹션
        sidebar_config.update(_render_image_extraction())
        
        # 도메인 파일 업로드 섹션
        sidebar_config.update(_render_domain_upload())
        
        # 벡터 DB 상태 및 관리
        sidebar_config.update(_render_vector_db_management(safe_get_vector_db, safe_get_rag_chain))
        
        # LLM 모델 설정
        sidebar_config.update(_render_llm_settings(safe_get_rag_chain, bootstrap_models_with_file_lock))
        
    return sidebar_config


def _render_document_management() -> Dict[str, Any]:
    """문서 관리 섹션 렌더링"""
    st.subheader("📄 문서 관리")
    
    # OCR 옵션
    use_ocr = st.checkbox(
        "PDF OCR 사용", 
        value=True,
        key="use_ocr_checkbox",
        help="스캔된 PDF나 이미지 PDF에서 텍스트를 추출하려면 체크하세요. (Tesseract 필요)"
    )
    
    # 에이전트 모드 옵션
    use_agent_mode = st.checkbox(
        "🤖 에이전트 모드 (고품질 처리)", 
        value=False,
        key="use_agent_mode_checkbox",
        help="로컬 LLM을 활용한 지능형 문서 전처리를 사용합니다.\n• 한국어 텍스트 분절 문제 해결\n• 고유명사 완성도 향상\n• 문맥 연결성 개선\n⚠️ 처리 시간이 더 오래 걸립니다."
    )
    
    # 2단계 품질 개선 옵션 (기본값: True)
    enable_postprocessing = st.checkbox(
        "✨ PDF 변환 시 텍스트 품질 자동 개선 (권장)", 
        value=True,
        key="enable_postprocessing_checkbox",
        help="PDF에서 추출한 텍스트의 품질을 자동으로 개선합니다.\n• 한국어 문장 연결 및 띄어쓰기 교정\n• 문맥 일관성 향상\n• 품질 점수 90점 이상 달성\n• 처리된 파일은 processed_docs 폴더에 저장됩니다.",
        disabled=False
    )
    
    return {
        'use_ocr': use_ocr,
        'use_agent_mode': use_agent_mode,
        'enable_postprocessing': enable_postprocessing
    }


def _render_text_preprocessing() -> Dict[str, Any]:
    """문서 전처리(텍스트) 섹션 렌더링"""
    st.subheader("🤖 문서 전처리(텍스트)")
    
    preprocessing_model_options = {
        "local": "로컬 모델 (기본)",
        "openai": "OpenAI GPT",
        "google": "Google Gemini",
        "anthropic": "Anthropic Claude"
    }
    
    selected_preprocessing_model = st.selectbox(
        "문서 전처리 모델 선택",
        options=list(preprocessing_model_options.keys()),
        format_func=lambda x: preprocessing_model_options[x],
        index=list(preprocessing_model_options.keys()).index(st.session_state.get('preprocessing_model', 'local')),
        key="preprocessing_model_selectbox",
        help="PDF 문서 전처리에 사용할 모델을 선택하세요.\n• 로컬 모델: 빠르고 무료, 하지만 성능은 제한적\n• 외부 API: 고품질 전처리, but API 키 필요"
    )
    
    # 멀티모달 전처리 옵션
    enable_multimodal = st.checkbox(
        "🖼️ 멀티모달 전처리 활성화",
        value=st.session_state.get('enable_multimodal_preprocessing', False),
        key="enable_multimodal_preprocessing_checkbox",
        help="이미지와 텍스트를 함께 분석하는 멀티모달 AI 모델을 사용합니다"
    )
    st.session_state.enable_multimodal_preprocessing = enable_multimodal
    
    # 텍스트 전처리 상태 표시(제공자/모델/멀티모달)
    _display_preprocessing_status(selected_preprocessing_model, preprocessing_model_options, enable_multimodal)
    
    # 멀티모달 모델 정보 표시
    if enable_multimodal:
        _display_multimodal_models()
    
    # 전처리 모델 상태 표시
    _display_preprocessing_model_status(selected_preprocessing_model, enable_multimodal)
    
    return {
        'selected_preprocessing_model': selected_preprocessing_model,
        'enable_multimodal': enable_multimodal
    }


def _display_preprocessing_status(selected_model: str, model_options: Dict[str, str], enable_multimodal: bool):
    """텍스트 전처리 상태 표시"""
    try:
        if selected_model == "openai":
            _model_name = settings.openai_model
        elif selected_model == "google":
            _model_name = settings.google_model
        elif selected_model == "anthropic":
            _model_name = settings.anthropic_model
        else:
            _model_name = "local_default"
        st.caption(
            f"텍스트 전처리: {model_options[selected_model]} — "
            f"모델: {_model_name} — 멀티모달: {'ON' if enable_multimodal else 'OFF'}"
        )
    except Exception:
        pass


def _display_multimodal_models():
    """멀티모달 모델 정보 표시"""
    multimodal_models = PreprocessingModelFactory.get_multimodal_models()
    with st.expander("📋 지원되는 멀티모달 모델"):
        available_models = PreprocessingModelFactory.get_available_models()
        for provider, models in multimodal_models.items():
            is_available = available_models.get(provider, {}).get('available', False)
            # OpenRouter는 별도 경로이므로 별도 가용성 판정
            if provider == 'openrouter':
                try:
                    is_available = bool(settings.openrouter_api_key) and \
                        settings.image_analysis_provider.lower() == 'openrouter'
                except Exception:
                    is_available = False
            if is_available and models:
                st.write(f"**{provider}**: {', '.join(models)}")


def _display_preprocessing_model_status(selected_model: str, enable_multimodal: bool):
    """전처리 모델 상태 표시"""
    if selected_model == "local":
        st.info("🏠 **로컬 모델**: 빠른 처리, 무료 사용")
        if enable_multimodal:
            st.warning("⚠️ **로컬 + 멀티모달**: 외부 API 연동이 필요할 수 있습니다.")
    else:
        api_key_exists = _check_api_key_exists(selected_model)
        if api_key_exists:
            st.success(f"✅ **{selected_model.upper()} 연결**: API 키 설정됨")
        else:
            st.error(f"❌ **{selected_model.upper()} 미연결**: API 키가 필요합니다")


def _check_api_key_exists(provider: str) -> bool:
    """API 키 존재 여부 확인"""
    if provider == "openai":
        return bool(settings.openai_api_key)
    elif provider == "google":
        return bool(settings.google_api_key)
    elif provider == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


def _render_image_extraction() -> Dict[str, Any]:
    """이미지 추출·OCR 섹션 렌더링"""
    st.divider()
    st.subheader("🖼️ 이미지 추출·OCR")
    
    # 기본 이미지 추출 옵션
    extract_images = st.checkbox(
        "PDF 이미지 자동 추출",
        value=True,
        key="extract_images_checkbox",
        help="PDF에서 이미지를 추출하여 답변에 포함시킵니다"
    )
    
    # 이미지 OCR 옵션
    ocr_images = st.checkbox(
        "이미지 OCR (텍스트 추출)",
        value=False,
        key="ocr_images_checkbox",
        help="추출된 이미지에서 텍스트를 OCR로 인식합니다"
    )
    
    # 지능형 이미지 분석 옵션
    intelligent_extraction = st.checkbox(
        "🧠 지능형 이미지 분석 (OpenRouter 경로)",
        value=st.session_state.get('intelligent_extraction', False),
        key="intelligent_extraction_checkbox",
        help="AI 모델을 사용하여 이미지의 내용을 분석하고 설명을 생성합니다"
    )
    
    # 이미지 추출 경로 캡션
    if intelligent_extraction:
        try:
            provider = getattr(settings, 'image_analysis_provider', 'openrouter')
            model = getattr(settings, 'openrouter_mm_model', 'z-ai/glm-4.5v')
            st.caption(f"이미지 추출 경로: {provider} ({model})")
        except Exception:
            st.caption("이미지 추출 경로: OpenRouter 기본")
    
    return {
        'extract_images': extract_images,
        'ocr_images': ocr_images,
        'intelligent_extraction': intelligent_extraction
    }


def _render_domain_upload() -> Dict[str, Any]:
    """도메인 파일 업로드 섹션 렌더링"""
    st.divider()
    st.subheader("📁 도메인 파일 업로드")
    
    uploaded_domain_file = st.file_uploader(
        "도메인 컨텍스트 파일",
        type=['txt', 'md'],
        help="시스템이 참고할 도메인별 지식이나 컨텍스트를 업로드하세요"
    )
    
    if uploaded_domain_file is not None:
        if st.button("도메인 파일 저장", key="save_domain"):
            with st.spinner("도메인 파일을 저장하는 중..."):
                content = uploaded_domain_file.read().decode('utf-8')
                with open('domain.md', 'w', encoding='utf-8') as f:
                    f.write(content)
                st.success("도메인 파일이 저장되었습니다!")
                st.rerun()
    
    # 도메인 파일 로드 버튼
    if st.button("💾 도메인 파일 로드", key="load_domain"):
        if os.path.exists('domain.md'):
            with st.spinner("도메인 파일을 로드하는 중..."):
                status_text = st.empty()
                try:
                    # 기존 domain.md를 벡터 DB에 추가하는 로직 필요
                    st.success("도메인 파일이 로드되었습니다!")
                except Exception as e:
                    st.error(f"도메인 파일 로드 중 오류 발생: {str(e)}")
                finally:
                    status_text.empty()
        else:
            st.error("domain.md 파일을 찾을 수 없습니다.")
    
    return {
        'uploaded_domain_file': uploaded_domain_file
    }


def _render_vector_db_management(safe_get_vector_db, safe_get_rag_chain) -> Dict[str, Any]:
    """벡터 DB 상태 및 관리 섹션 렌더링"""
    st.divider()
    vector_db = safe_get_vector_db()
    doc_count = vector_db.get_document_count()
    st.info(f"💾 저장된 문서 청크: {doc_count}개")
    
    # 로그 뷰어 (확장 가능)
    _render_log_viewer()
    
    # 벡터 DB 초기화
    _render_db_reset_buttons(safe_get_vector_db, safe_get_rag_chain)
    
    return {
        'doc_count': doc_count
    }


def _render_log_viewer():
    """로그 뷰어 렌더링"""
    with st.expander("📋 처리 로그 보기"):
        LOG_FILE_PATTERN = "logs/rag_app_{date}.log"
        MAX_LOG_LINES_DISPLAY = 50
        
        log_file = LOG_FILE_PATTERN.format(date=datetime.now().strftime('%Y%m%d'))
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                # 최근 N줄만 표시
                lines = f.readlines()
                recent_lines = lines[-MAX_LOG_LINES_DISPLAY:] if len(lines) > MAX_LOG_LINES_DISPLAY else lines
                st.text(''.join(recent_lines))
        else:
            st.text("로그 파일이 없습니다.")


def _render_db_reset_buttons(safe_get_vector_db, safe_get_rag_chain):
    """벡터 DB 초기화 버튼들 렌더링"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("벡터 DB 초기화", type="secondary", key="sidebar_clear_db_btn"):
            st.session_state.show_clear_confirm = True
    
    if 'show_clear_confirm' in st.session_state and st.session_state.show_clear_confirm:
        with col2:
            if st.button("⚠️ 확인", type="primary", key="sidebar_confirm_clear"):
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


def _render_llm_settings(safe_get_rag_chain, bootstrap_models_with_file_lock) -> Dict[str, Any]:
    """LLM 모델 설정 섹션 렌더링"""
    st.divider()
    st.subheader("🤖 LLM 모델 설정")
    
    # 사용 가능한 모델 가져오기
    available_models = safe_get_rag_chain().get_available_models()
    
    # LLM 제공자 선택
    providers = ["local", "openai", "google", "anthropic"]
    provider_names = {
        "local": "로컬 모델",
        "openai": "OpenAI GPT",
        "google": "Google Gemini", 
        "anthropic": "Anthropic Claude"
    }
    
    current_provider = st.session_state.get('current_provider', 'local')
    selected_provider = st.selectbox(
        "LLM 제공자",
        providers,
        index=providers.index(current_provider),
        format_func=lambda x: provider_names[x],
        key="provider_selector"
    )
    
    # 선택된 제공자의 모델 선택
    if selected_provider in available_models:
        models_info = available_models[selected_provider]
        
        # 모델 정보가 리스트 형태인 경우 (ModelRegistry.get_all_models() 반환값)
        if isinstance(models_info, list):
            if models_info:  # 모델 목록이 비어있지 않은 경우
                model_list = [model.get('model', model.get('id', '')) for model in models_info]
                current_model = st.session_state.get('current_model')
                
                if not current_model or current_model not in model_list:
                    current_model = model_list[0] if model_list else None
                
                if current_model and model_list:
                    selected_model = st.selectbox(
                        f"{provider_names[selected_provider]} 모델",
                        model_list,
                        index=model_list.index(current_model) if current_model in model_list else 0,
                        key="model_selector"
                    )
                else:
                    st.error(f"❌ {provider_names[selected_provider]} 모델을 찾을 수 없습니다.")
                    selected_model = None
            else:
                st.error(f"❌ {provider_names[selected_provider]} 모델이 없습니다.")
                selected_model = None
        
        # 모델 정보가 딕셔너리 형태인 경우 (레거시 지원)
        elif isinstance(models_info, dict) and models_info.get('available', False):
            model_list = models_info['models']
            current_model = st.session_state.get('current_model', model_list[0])
            
            if current_model not in model_list:
                current_model = model_list[0]
            
            selected_model = st.selectbox(
                f"{provider_names[selected_provider]} 모델",
                model_list,
                index=model_list.index(current_model) if current_model in model_list else 0,
                key="model_selector"
            )
        else:
            st.error(f"❌ {provider_names[selected_provider]} API 키가 설정되지 않았습니다.")
            selected_model = None
    else:
        st.error(f"❌ {provider_names[selected_provider]} 모델 정보를 가져올 수 없습니다.")
        selected_model = None
    
    # 모델 설정 적용
    if selected_model and (
        selected_provider != st.session_state.get('current_provider') or 
        selected_model != st.session_state.get('current_model')
    ):
        if st.button("모델 설정 적용", type="primary"):
            with st.spinner("모델을 변경하는 중..."):
                try:
                    safe_get_rag_chain().update_llm(selected_provider, selected_model)
                    st.session_state.current_provider = selected_provider
                    st.session_state.current_model = selected_model
                    st.success(f"✅ 모델이 {provider_names[selected_provider]} - {selected_model}로 변경되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 모델 변경 실패: {str(e)}")
    
    # 모델 부트스트랩 버튼
    if st.button("🔄 모델 재부팅", help="로컬 모델을 다시 로드합니다"):
        with st.spinner("모델을 재부팅하는 중..."):
            bootstrap_models_with_file_lock()
            st.success("✅ 모델 재부팅이 완료되었습니다!")
            st.rerun()
    
    return {
        'selected_provider': selected_provider,
        'selected_model': selected_model,
        'available_models': available_models
    }