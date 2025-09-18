"""
채팅 인터페이스 컴포넌트
"""

import streamlit as st
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.rag.rag_chain import RAGChain


def render_chat_interface(
    rag_chain: RAGChain,
    display_images_in_response,
    sidebar_config: Dict[str, Any]
) -> None:
    """
    메인 채팅 인터페이스를 렌더링합니다.
    
    Args:
        rag_chain: RAG 체인 인스턴스
        display_images_in_response: 이미지 표시 함수
        sidebar_config: 사이드바 설정값들
    """
    
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    _display_chat_history(display_images_in_response)
    
    # 채팅 입력 및 처리
    _handle_chat_input(rag_chain, display_images_in_response, sidebar_config)


def _display_chat_history(display_images_in_response) -> None:
    """채팅 히스토리를 표시합니다."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "context_documents" in message:
                # 이미지가 포함된 응답 처리
                processed_content = display_images_in_response(
                    message["content"], 
                    message.get("context_documents", [])
                )
                st.markdown(processed_content, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])


def _handle_chat_input(
    rag_chain: RAGChain,
    display_images_in_response,
    sidebar_config: Dict[str, Any]
) -> None:
    """채팅 입력을 처리합니다."""
    
    # 채팅 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 어시스턴트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    # RAG 체인으로 응답 생성
                    start_time = time.time()
                    response_data = rag_chain.invoke(prompt)
                    end_time = time.time()
                    
                    # 응답 처리
                    if isinstance(response_data, dict):
                        response_text = response_data.get('answer', str(response_data))
                        context_documents = response_data.get('context_documents', [])
                        metadata = response_data.get('metadata', {})
                    else:
                        response_text = str(response_data)
                        context_documents = []
                        metadata = {}
                    
                    # 이미지 표시 처리
                    if sidebar_config.get('extract_images', True):
                        processed_response = display_images_in_response(response_text, context_documents)
                        st.markdown(processed_response, unsafe_allow_html=True)
                    else:
                        st.markdown(response_text)
                    
                    # 성능 정보 표시
                    processing_time = end_time - start_time
                    _display_performance_info(processing_time, metadata, context_documents)
                    
                    # 어시스턴트 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "context_documents": context_documents,
                        "metadata": metadata,
                        "processing_time": processing_time
                    })
                    
                except Exception as e:
                    error_message = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })


def _display_performance_info(
    processing_time: float,
    metadata: Dict[str, Any],
    context_documents: List[Any]
) -> None:
    """성능 정보를 표시합니다."""
    
    with st.expander("📊 응답 정보", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("처리 시간", f"{processing_time:.2f}초")
        
        with col2:
            st.metric("참조 문서", f"{len(context_documents)}개")
        
        with col3:
            if 'total_tokens' in metadata:
                st.metric("사용 토큰", f"{metadata['total_tokens']}")
            else:
                st.metric("모델", metadata.get('model', 'Unknown'))
        
        # 컨텍스트 문서 정보
        if context_documents:
            st.write("**참조된 문서:**")
            for i, doc in enumerate(context_documents[:3], 1):  # 최대 3개만 표시
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    st.write(f"{i}. {source} (페이지: {page})")
                else:
                    st.write(f"{i}. 문서 {i}")


def render_chat_controls() -> None:
    """채팅 제어 버튼들을 렌더링합니다."""
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ 채팅 기록 삭제", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 채팅 내보내기", key="export_chat"):
            _export_chat_history()
    
    with col3:
        st.caption(f"총 {len(st.session_state.get('messages', []))}개의 메시지")


def _export_chat_history() -> None:
    """채팅 기록을 내보냅니다."""
    
    if not st.session_state.get('messages'):
        st.warning("내보낼 채팅 기록이 없습니다.")
        return
    
    # 채팅 기록을 텍스트로 변환
    chat_text = f"# 채팅 기록 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, message in enumerate(st.session_state.messages, 1):
        role = "사용자" if message["role"] == "user" else "어시스턴트"
        content = message["content"]
        
        chat_text += f"## {i}. {role}\n\n{content}\n\n"
        
        # 성능 정보 추가 (어시스턴트 메시지인 경우)
        if message["role"] == "assistant" and "processing_time" in message:
            processing_time = message["processing_time"]
            context_count = len(message.get("context_documents", []))
            chat_text += f"*처리 시간: {processing_time:.2f}초, 참조 문서: {context_count}개*\n\n"
        
        chat_text += "---\n\n"
    
    # 다운로드 버튼 제공
    st.download_button(
        label="📥 텍스트 파일로 다운로드",
        data=chat_text.encode('utf-8'),
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="download_chat"
    )
    
    st.success("채팅 기록을 다운로드할 수 있습니다.")


def render_feedback_interface() -> None:
    """피드백 인터페이스를 렌더링합니다."""
    
    with st.expander("💬 피드백 보내기"):
        st.write("답변의 품질을 평가해주세요:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 도움이 되었어요", key="feedback_positive"):
                st.success("긍정적인 피드백 감사합니다!")
                # TODO: 피드백을 데이터베이스에 저장하여 모델 개선에 활용
        
        with col2:
            if st.button("👎 개선이 필요해요", key="feedback_negative"):
                st.info("피드백 감사합니다. 더 나은 답변을 제공하도록 노력하겠습니다!")
                # TODO: 피드백을 데이터베이스에 저장하여 모델 개선에 활용
        
        # 상세 피드백
        detailed_feedback = st.text_area(
            "상세한 피드백 (선택사항)",
            placeholder="답변에 대한 구체적인 의견이나 개선 사항을 알려주세요...",
            key="detailed_feedback"
        )
        
        if st.button("피드백 제출", key="submit_feedback") and detailed_feedback:
            st.success("상세한 피드백이 제출되었습니다. 감사합니다!")
            # TODO: 상세 피드백을 데이터베이스에 저장