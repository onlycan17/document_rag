"""
파일 업로드 컴포넌트
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.loaders.document_loader import DocumentLoader
from src.utils.document_processor import DocumentProcessor
from src.constants import (
    TEMP_DOCUMENT_PATH,
    LOG_FILE_PATTERN,
    MAX_LOG_LINES_DISPLAY
)


def render_file_upload_interface(
    safe_get_vector_db,
    preprocessing_settings: Dict[str, Any]
) -> None:
    """
    파일 업로드 인터페이스를 렌더링합니다.
    
    Args:
        safe_get_vector_db: 벡터 DB 가져오기 함수
        preprocessing_settings: 전처리 설정 딕셔너리
    """
    
    # 전처리 설정 표시
    _display_preprocessing_settings(preprocessing_settings)
    
    # 파일 업로드 위젯
    uploaded_files = st.file_uploader(
        "문서 업로드 (TXT, MD, PDF, DOCX) 🆕 이미지 추출 지원",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="PDF/DOCX 파일의 이미지도 자동 추출됩니다"
    )
    
    # 파일 처리 버튼 및 로직
    if uploaded_files:
        _handle_file_upload(uploaded_files, safe_get_vector_db, preprocessing_settings)
    
    # 기존 문서 로드 버튼
    _render_domain_file_loader(safe_get_vector_db)
    
    # 벡터 DB 상태 및 관리
    _render_vector_db_status(safe_get_vector_db)


def _display_preprocessing_settings(preprocessing_settings: Dict[str, Any]) -> None:
    """전처리 설정 상태를 표시합니다."""
    
    if preprocessing_settings.get('use_agent_mode', False):
        st.info("🤖 **에이전트 모드 활성화**: 고품질 전처리 사용 중")
    
    if preprocessing_settings.get('enable_postprocessing', False):
        st.info("✨ **2단계 품질 개선 활성화**: PDF 변환 후 자동으로 텍스트 품질을 개선합니다.")
    
    st.divider()


def _handle_file_upload(
    uploaded_files: List[Any],
    safe_get_vector_db,
    preprocessing_settings: Dict[str, Any]
) -> None:
    """파일 업로드 처리를 담당합니다."""
    
    if st.button("문서 처리 및 저장"):
        # DocumentLoader 재초기화
        _reinitialize_document_loader(preprocessing_settings)
        
        # 디렉토리 준비
        DocumentProcessor.prepare_directories()
        
        # 파일 처리 실행
        _process_uploaded_files(uploaded_files, safe_get_vector_db, preprocessing_settings)


def _reinitialize_document_loader(preprocessing_settings: Dict[str, Any]) -> None:
    """설정 변경에 따라 DocumentLoader를 재초기화합니다."""
    
    current_loader = st.session_state.document_loader
    
    # 설정 변경 감지
    settings_changed = (
        preprocessing_settings.get('use_agent_mode', False) != current_loader.use_agent_preprocessing or
        preprocessing_settings.get('enable_postprocessing', False) != getattr(current_loader, 'enable_postprocessing', False) or
        preprocessing_settings.get('use_intelligent_extraction', False) != getattr(current_loader, 'use_intelligent_image_extraction', False) or
        preprocessing_settings.get('selected_preprocessing_model', 'local') != getattr(current_loader, 'preprocessing_model', 'local')
    )

    if settings_changed:
        st.session_state.document_loader = DocumentLoader(
            use_ocr=current_loader.use_ocr,
            use_agent_preprocessing=preprocessing_settings.get('use_agent_mode', False),
            enable_postprocessing=preprocessing_settings.get('enable_postprocessing', False),
            use_intelligent_image_extraction=preprocessing_settings.get('use_intelligent_extraction', False),
            preprocessing_model=preprocessing_settings.get('selected_preprocessing_model', 'local'),
            enable_multimodal_preprocessing=st.session_state.enable_multimodal_preprocessing
        )


def _process_uploaded_files(
    uploaded_files: List[Any],
    safe_get_vector_db,
    preprocessing_settings: Dict[str, Any]
) -> None:
    """업로드된 파일들을 처리합니다."""
    
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
        # 개별 파일 처리
        file_stats = _process_single_file(
            uploaded_file, 
            file_idx, 
            total_files, 
            overall_progress, 
            overall_status,
            safe_get_vector_db,
            preprocessing_settings
        )
        
        # 통계 업데이트
        if file_stats['success']:
            processed_files += 1
            total_chunks += file_stats['chunks']
        else:
            failed_files += 1
        total_processing_time += file_stats['processing_time']
    
    # 전체 처리 완료 후 결과 표시
    _display_processing_summary(
        overall_progress, 
        overall_status,
        total_files, 
        processed_files, 
        failed_files, 
        total_chunks, 
        total_processing_time,
        safe_get_vector_db
    )


def _process_single_file(
    uploaded_file: Any,
    file_idx: int,
    total_files: int,
    overall_progress: Any,
    overall_status: Any,
    safe_get_vector_db,
    preprocessing_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """개별 파일을 처리합니다."""
    
    # 진행 상황 업데이트
    overall_status.text(f"파일 {file_idx}/{total_files} 처리 중: {uploaded_file.name}")
    overall_progress.progress(file_idx / total_files)
    
    # 파일 정보 표시
    _display_file_info(uploaded_file, file_idx, total_files)
    
    # 진행 상황 위젯
    file_progress = st.progress(0)
    file_status = st.empty()
    
    # 임시 파일 저장
    temp_path = TEMP_DOCUMENT_PATH.format(filename=uploaded_file.name)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # 진행 상황 콜백
    def update_progress(progress, message):
        file_progress.progress(progress)
        file_status.text(message)
    
    # 문서 처리 실행
    processing_result = _execute_document_processing(
        uploaded_file,
        temp_path,
        update_progress,
        safe_get_vector_db,
        preprocessing_settings
    )
    
    # 결과 표시
    _display_file_processing_result(uploaded_file, processing_result)
    
    # 정리
    file_progress.empty()
    file_status.empty()
    
    try:
        os.remove(temp_path)
    except:
        pass
    
    return processing_result


def _display_file_info(uploaded_file: Any, file_idx: int, total_files: int) -> None:
    """파일 정보를 표시합니다."""
    
    st.divider()
    file_col1, file_col2 = st.columns([3, 1])
    
    with file_col1:
        st.write(f"**📄 [{file_idx}/{total_files}] {uploaded_file.name}**")
    
    with file_col2:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"{file_size_mb:.1f}MB")


def _execute_document_processing(
    uploaded_file: Any,
    temp_path: str,
    update_progress,
    safe_get_vector_db,
    preprocessing_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """문서 처리를 실행합니다."""
    
    start_time = time.time()
    processing_error = None
    documents = None
    processing_time = 0.0
    
    try:
        # DocumentLoader 설정 업데이트
        _update_document_loader_settings(preprocessing_settings)
        
        # 벡터 DB에 추가하기 전 청크 수 확인
        before_count = safe_get_vector_db().get_document_count()
        
        # 문서 로드
        documents = st.session_state.document_loader.load_document(temp_path, update_progress)
        
        # 지능형 이미지 추출 결과 확인
        image_extraction_result = _check_image_extraction_result(
            uploaded_file,
            preprocessing_settings.get('use_intelligent_extraction', False)
        )
        
        # 문서 처리 및 저장
        if documents or image_extraction_result['success']:
            saved_chunks = 0
            if documents:
                update_progress(0.95, "벡터 데이터베이스에 저장 중...")
                safe_get_vector_db().add_documents(documents)
                after_count = safe_get_vector_db().get_document_count()
                saved_chunks = after_count - before_count
            
            processing_time = time.time() - start_time
            update_progress(1.0, "완료!")
            
            # 청크 분석 및 메타데이터
            chunk_analysis = analyze_chunks(documents) if documents else None
            pdf_metadata = get_pdf_metadata(documents) if documents else None
            
            return {
                'success': True,
                'chunks': len(documents) if documents else 0,
                'processing_time': processing_time,
                'chunk_analysis': chunk_analysis,
                'pdf_metadata': pdf_metadata,
                'image_extraction': image_extraction_result,
                'saved_chunks': saved_chunks,
                'error': None
            }
        else:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'chunks': 0,
                'processing_time': processing_time,
                'error': "문서를 로드할 수 없음"
            }
    
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'success': False,
            'chunks': 0,
            'processing_time': processing_time,
            'error': str(e)
        }


def _update_document_loader_settings(preprocessing_settings: Dict[str, Any]) -> None:
    """DocumentLoader 설정을 업데이트합니다."""
    
    current_loader = st.session_state.document_loader
    
    # 설정 확인 및 업데이트
    if (getattr(current_loader, 'use_ocr', True) != preprocessing_settings.get('use_ocr', True) or
        getattr(current_loader, 'use_agent_preprocessing', False) != preprocessing_settings.get('use_agent_mode', False) or
        getattr(current_loader, 'use_intelligent_image_extraction', False) != preprocessing_settings.get('use_intelligent_extraction', False)):
        
        st.session_state.document_loader = DocumentLoader(
            use_ocr=preprocessing_settings.get('use_ocr', True),
            use_agent_preprocessing=preprocessing_settings.get('use_agent_mode', False),
            use_intelligent_image_extraction=preprocessing_settings.get('use_intelligent_extraction', False),
            preprocessing_model=preprocessing_settings.get('selected_preprocessing_model', 'local')
        )


def _check_image_extraction_result(uploaded_file: Any, use_intelligent_extraction: bool) -> Dict[str, Any]:
    """지능형 이미지 추출 결과를 확인합니다."""
    
    if not use_intelligent_extraction:
        return {'success': False, 'count': 0}
    
    pdf_name = Path(uploaded_file.name).stem
    images_dir = Path(f"data/extracted_images/{pdf_name}/images")
    
    if images_dir.exists():
        extracted_images = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        extracted_count = len(extracted_images)
        return {'success': extracted_count > 0, 'count': extracted_count}
    
    return {'success': False, 'count': 0}


def _display_file_processing_result(uploaded_file: Any, result: Dict[str, Any]) -> None:
    """파일 처리 결과를 표시합니다."""
    
    if result['success']:
        # 성공 메시지
        success_col1, success_col2 = st.columns([2, 1])
        
        with success_col1:
            st.success(f"✅ {uploaded_file.name} 처리 완료")
            _display_processing_info(result)
        
        with success_col2:
            st.metric("처리 시간", f"{result['processing_time']:.1f}초")
        
        # 청크 분석 정보 표시
        if result.get('chunk_analysis'):
            _display_chunk_analysis(result['chunk_analysis'])
        
        # 품질 검증 결과 표시
        _display_quality_verification(result)
        
        # 검색 품질 테스트 (생략 - 추후 구현 예정)
    
    else:
        # 실패 메시지
        st.error(f"❌ {uploaded_file.name} 처리 실패: {result['error']}")
        st.caption(f"처리 시간: {result['processing_time']:.1f}초")


def _display_processing_info(result: Dict[str, Any]) -> None:
    """처리 정보를 표시합니다."""
    
    # 지능형 이미지 추출만 성공한 경우
    if result.get('image_extraction', {}).get('success') and result['chunks'] == 0:
        st.caption(f"**처리 정보:** 지능형 이미지 추출 | {result['image_extraction']['count']}개 이미지 추출")
    
    # PDF 메타데이터가 있는 경우
    elif result.get('pdf_metadata'):
        pdf_metadata = result['pdf_metadata']
        info_text = f"**처리 정보:** {pdf_metadata['extraction_method']}"
        
        if pdf_metadata.get('page_count'):
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


def _display_chunk_analysis(chunk_analysis: Dict[str, Any]) -> None:
    """청크 분석 정보를 표시합니다."""
    
    chunk_col1, chunk_col2, chunk_col3, chunk_col4 = st.columns(4)
    
    with chunk_col1:
        st.metric("청크 수", f"{chunk_analysis['count']}개")
    
    with chunk_col2:
        st.metric("평균 크기", f"{chunk_analysis['avg_size']:.0f}자")
    
    with chunk_col3:
        st.metric("최소 크기", f"{chunk_analysis['min_size']}자")
    
    with chunk_col4:
        st.metric("최대 크기", f"{chunk_analysis['max_size']}자")


def _display_quality_verification(result: Dict[str, Any]) -> None:
    """품질 검증 결과를 표시합니다."""
    
    if result['chunks'] > 0:
        if result['chunks'] == result.get('saved_chunks', 0):
            st.info(f"💾 품질 검증: 모든 청크({result['chunks']}개)가 성공적으로 저장됨")
        else:
            st.warning(f"⚠️ 품질 검증: 로드된 청크({result['chunks']}개) vs 저장된 청크({result.get('saved_chunks', 0)}개)")
    
    elif result.get('image_extraction', {}).get('success'):
        st.info(f"🖼️ 이미지 추출: {result['image_extraction']['count']}개 이미지가 성공적으로 추출됨")


# _display_search_quality_test 함수는 제거됨 (추후 구현 예정)


def _display_processing_summary(
    overall_progress: Any,
    overall_status: Any,
    total_files: int,
    processed_files: int,
    failed_files: int,
    total_chunks: int,
    total_processing_time: float,
    safe_get_vector_db
) -> None:
    """전체 처리 결과 요약을 표시합니다."""
    
    # 진행 표시 제거
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
    time.sleep(2)
    st.rerun()


def _render_domain_file_loader(safe_get_vector_db) -> None:
    """기존 domain.md 파일 로드 버튼을 렌더링합니다."""
    
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


def _render_vector_db_status(safe_get_vector_db) -> None:
    """벡터 DB 상태 및 관리를 렌더링합니다."""
    
    st.divider()
    
    # 벡터 DB 상태 표시
    vector_db = safe_get_vector_db()
    doc_count = vector_db.get_document_count()
    st.info(f"💾 저장된 문서 청크: {doc_count}개")
    
    # 로그 뷰어
    _render_log_viewer()
    
    # 벡터 DB 초기화
    _render_vector_db_clear_controls(safe_get_vector_db)


def _render_log_viewer() -> None:
    """로그 뷰어를 렌더링합니다."""
    
    with st.expander("📋 처리 로그 보기"):
        log_file = LOG_FILE_PATTERN.format(date=datetime.now().strftime('%Y%m%d'))
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                recent_lines = lines[-MAX_LOG_LINES_DISPLAY:] if len(lines) > MAX_LOG_LINES_DISPLAY else lines
                st.text(''.join(recent_lines))
        else:
            st.text("로그 파일이 없습니다.")


def _render_vector_db_clear_controls(safe_get_vector_db) -> None:
    """벡터 DB 초기화 컨트롤을 렌더링합니다."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("벡터 DB 초기화", type="secondary", key="file_clear_db_btn"):
            st.session_state.show_clear_confirm = True
    
    if 'show_clear_confirm' in st.session_state and st.session_state.show_clear_confirm:
        with col2:
            if st.button("⚠️ 확인", type="primary", key="file_confirm_clear"):
                safe_get_vector_db().clear_database()
                # RAG 체인도 재초기화
                from src.rag.rag_chain import RAGChain
                st.session_state.rag_chain = RAGChain(
                    provider=st.session_state.current_provider, 
                    model=st.session_state.current_model,
                    vector_db=safe_get_vector_db()
                )
                st.success("벡터 데이터베이스가 초기화되었습니다.")
                st.session_state.show_clear_confirm = False


def analyze_chunks(documents: List[Any]) -> Optional[Dict[str, Any]]:
    """
    업로드된 문서의 청크 분석
    
    Args:
        documents: 분석할 문서 청크 리스트
        
    Returns:
        청크 분석 결과 딕셔너리 또는 None (문서가 없는 경우)
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
        documents: 문서 리스트
        
    Returns:
        PDF 메타데이터 딕셔너리 또는 None
    """
    if not documents or not hasattr(documents[0], 'metadata'):
        return None
    
    metadata = documents[0].metadata
    return {
        'extraction_method': metadata.get('extraction_method', 'Unknown'),
        'page_count': metadata.get('page_count'),
        'processing_method': metadata.get('processing_method', ''),
        'image_count': metadata.get('image_count', 0)
    }