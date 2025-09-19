"""
채팅 인터페이스 컴포넌트
"""

import streamlit as st
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.rag.rag_chain import RAGChain
from ui.services.app_service import AppService


def render_chat_interface(
    rag_chain: RAGChain,
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
    _display_chat_history()

    # 채팅 입력 및 처리
    _handle_chat_input(rag_chain, sidebar_config)


def _display_chat_history() -> None:
    """채팅 히스토리를 표시합니다."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "context_documents" in message:
                # 텍스트는 Markdown으로 표시하고, 이미지들은 st.image로 직접 렌더링합니다.
                st.markdown(message["content"])
                try:
                    images_info = AppService.extract_images_from_content(message.get("content", ""), message.get("context_documents", []))
                    for img in images_info:
                        path = img.get('path')
                        caption = f"파일: {img.get('filename', 'Unknown')} | 출처: {img.get('source', 'Unknown')}"
                        try:
                            if path and os.path.exists(path):
                                st.image(path, caption=caption, use_column_width=False)
                        except Exception:
                            # 안전하게 무시하고 계속 렌더링
                            pass
                except Exception:
                    # 추출 실패시 기존 컨텐츠만 표시
                    pass
            else:
                st.markdown(message["content"])


def _handle_chat_input(
    rag_chain: RAGChain,
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
            # 스트리밍 우선 시도: rag_chain.stream_query가 존재하면 스트리밍을 사용하여 실시간 업데이트
            start_time = time.time()
            response_text = ""
            context_documents = []
            metadata = {}

            try:
                stream_gen = None
                # 스트리밍 메서드 존재 시 사용
                if hasattr(rag_chain, 'stream_query'):
                    try:
                        stream_gen = rag_chain.stream_query(prompt)
                    except Exception:
                        stream_gen = None

                # 스트리밍 사용 가능한 경우 제너레이터로 청크 처리
                if stream_gen is not None:
                    placeholder = st.empty()
                    status_placeholder = st.empty()

                    # stream_gen은 dict 청크를 yield 해야 함 (type, content 등)
                    for chunk in stream_gen:
                        try:
                            chunk_type = chunk.get('type', '')
                            # QueryEngine.stream_query yields types: 'status', 'content', 'complete', 'error'
                            if chunk_type == 'status':
                                status_placeholder.info(chunk.get('content', ''))
                            elif chunk_type == 'content':
                                # 청크별 부분 텍스트
                                chunk_content = chunk.get('content', '')
                                response_text += chunk_content
                                # 일부 스트리밍 구현은 full_content를 제공하므로 우선 순위로 사용
                                full = chunk.get('full_content') or response_text
                                placeholder.markdown(full)
                            elif chunk_type == 'complete':
                                # 스트리밍 완료: full_content와 sources 등의 메타 포함
                                response_text = chunk.get('full_content', response_text)
                                context_documents = chunk.get('sources', context_documents)
                                # search_info/metadata가 있는 경우 metadata 변수에 저장
                                if 'search_info' in chunk:
                                    metadata['search_info'] = chunk.get('search_info')
                                placeholder.markdown(response_text)
                                break
                            elif chunk_type == 'error':
                                # 오류 청크 수신시 표시하고 종료
                                err = chunk.get('content', '스트리밍 중 오류가 발생했습니다.')
                                status_placeholder.error(err)
                                response_text = ''
                                break
                        except Exception:
                            # 개별 청크 처리 중 에러는 무시하고 계속 스트리밍
                            continue

                    # 상태 정리
                    status_placeholder.empty()
                    end_time = time.time()
                else:
                    # 스트리밍을 사용할 수 없으면 기존 동기 방식으로 폴백
                    with st.spinner("답변을 생성하는 중..."):
                        response_data = rag_chain.invoke(prompt)
                        end_time = time.time()

                        if isinstance(response_data, dict):
                            response_text = response_data.get('answer', str(response_data))
                            context_documents = response_data.get('context_documents', [])
                            metadata = response_data.get('metadata', {})
                        else:
                            response_text = str(response_data)
                            context_documents = []
                            metadata = {}

                        st.markdown(response_text)

                # 이미지 및 추가 정보 렌더링
                if sidebar_config.get('extract_images', True) and response_text:
                    try:
                        images_info = AppService.extract_images_from_content(response_text, context_documents)
                        
                        if images_info:
                            for img in images_info:
                                path = img.get('path')
                                filename = img.get('filename', 'Unknown')
                                source = img.get('source', 'Unknown')
                                caption = f"파일: {filename} | 출처: {source}"
                                
                                try:
                                    if path and os.path.exists(path):
                                        st.image(path, caption=caption, use_column_width=False)
                                except Exception as e:
                                    st.warning(f"이미지 렌더링 중 오류 발생: {path} ({e})")
                    except Exception as e:
                        st.error(f"이미지 추출 과정에서 오류가 발생했습니다: {e}")

                # 성능 정보 표시 및 세션 저장
                processing_time = (end_time - start_time) if 'end_time' in locals() else time.time() - start_time
                _display_performance_info(processing_time, metadata, context_documents)

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