"""채팅 인터페이스 컴포넌트"""

from __future__ import annotations

import logging
import time
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.rag.rag_chain import RAGChain
from src.utils.source_highlight import render_source_preview
from ui.services.app_service import AppService
from ui.services.streaming_service import process_streaming_response

logger = logging.getLogger(__name__)


@dataclass
class AssistantResponse:
    """어시스턴트 응답 결과"""

    text: str = ""
    question: str = ""
    context_documents: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Dict[str, str]] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None
    text_rendered: bool = False


def render_chat_interface(rag_chain: RAGChain, sidebar_config: Dict[str, Any]) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "langfuse_session_id" not in st.session_state:
        # 브라우저 세션당 하나의 대화 ID — Langfuse에서 대화 단위 그룹핑에 사용
        st.session_state.langfuse_session_id = str(uuid4())
    _display_chat_history()
    _handle_chat_input(rag_chain, sidebar_config)


def _display_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message.get("content", ""))
            if message.get("role") == "assistant":
                _render_images(message.get("images", []))
                _render_source_documents(message.get("context_documents", []), message.get("question", ""))


def _handle_chat_input(rag_chain: RAGChain, sidebar_config: Dict[str, Any]) -> None:
    prompt = st.chat_input("질문을 입력하세요...")
    if not prompt:
        return
    _append_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = _generate_assistant_response(prompt, rag_chain, sidebar_config)
        _render_assistant_output(response)
    _store_assistant_message(response, prompt)


def _append_user_message(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})


def _generate_assistant_response(
    prompt: str,
    rag_chain: RAGChain,
    sidebar_config: Dict[str, Any],
) -> AssistantResponse:
    start_time = time.time()
    response = AssistantResponse(question=prompt)
    try:
        stream_gen = _get_stream_generator(rag_chain, prompt)
        if stream_gen is not None:
            response_placeholder = st.empty()
            status_placeholder = st.empty()
            stream_result = process_streaming_response(
                stream_gen,
                response_placeholder,
                status_placeholder,
                logger=logger,
            )
            response.text = stream_result.answer
            response.context_documents = stream_result.context_documents
            response.metadata = {"search_info": stream_result.search_info}
            response.text_rendered = stream_result.status != "error"
            if stream_result.status == "error":
                response.error = stream_result.error
        else:
            with st.spinner("답변을 생성하는 중..."):
                sync_result = rag_chain.invoke(prompt, session_id=st.session_state.langfuse_session_id)
            response.text, response.context_documents, response.metadata = _normalize_sync_result(sync_result)
        response.processing_time = time.time() - start_time
        if response.text and sidebar_config.get("extract_images", True):
            response.images = AppService.extract_images_from_content(
                response.text,
                response.context_documents,
                debug=sidebar_config.get("debug_mode", False),
            )
        return response
    except Exception as exc:
        logger.exception("어시스턴트 응답 생성 실패")
        response.error = f"답변 생성 중 오류가 발생했습니다: {exc}"
        response.text = response.error
        response.processing_time = time.time() - start_time
        return response


def _get_stream_generator(rag_chain: RAGChain, prompt: str):
    if not hasattr(rag_chain, "stream_query"):
        return None
    try:
        return rag_chain.stream_query(prompt, session_id=st.session_state.langfuse_session_id)
    except Exception:
        return None


def _normalize_sync_result(result: Any) -> tuple[str, List[Any], Dict[str, Any]]:
    if isinstance(result, dict):
        text = result.get("answer", "")
        documents = result.get("context_documents", [])
        metadata = result.get("metadata", {})
        return text or "", documents, metadata
    return str(result), [], {}


def _render_assistant_output(response: AssistantResponse) -> None:
    if not response.text_rendered:
        st.markdown(response.text)
    _render_images(response.images)
    if response.error:
        st.error(response.error)
        return
    _render_source_documents(response.context_documents, response.question)
    _display_performance_info(response.processing_time, response.metadata, response.context_documents)


def _render_source_documents(documents: List[Any], question: str) -> None:
    """출처 문서를 펼쳐보기 형태로 표시하고, 질문 관련 문단을 하이라이트해 먼저 보여준다."""
    if not documents:
        return
    with st.expander(f"📑 출처 보기 ({len(documents)}개)", expanded=False):
        for index, doc in enumerate(documents, start=1):
            metadata = getattr(doc, "metadata", {})
            file_name = metadata.get("file_name", metadata.get("source", "Unknown"))
            page = metadata.get("page", "N/A")
            header = f"[{index}] {file_name}"
            if page and str(page) != "N/A":
                header += f" (p.{page})"
            with st.expander(header, expanded=False):
                content = getattr(doc, "page_content", "")
                st.markdown(render_source_preview(content, question))
                with st.expander("전체 내용 보기"):
                    st.markdown(content)


def _render_images(images: List[Dict[str, str]]) -> None:
    for image in images:
        path = Path(image.get("path", ""))
        if not path.exists():
            continue
        caption = f"파일: {image.get('filename', 'Unknown')} | 출처: {image.get('source', 'Unknown')}"
        st.image(str(path), caption=caption, use_column_width=False)


def _store_assistant_message(response: AssistantResponse, question: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.text,
            "question": question,
            "context_documents": response.context_documents,
            "metadata": response.metadata,
            "processing_time": response.processing_time,
            "images": response.images,
        }
    )


def _display_performance_info(
    processing_time: float,
    metadata: Dict[str, Any],
    context_documents: List[Any],
) -> None:
    with st.expander("📊 응답 정보", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("처리 시간", f"{processing_time:.2f}초")
        col2.metric("참조 문서", f"{len(context_documents)}개")
        if "total_tokens" in metadata:
            col3.metric("사용 토큰", f"{metadata.get('total_tokens', 0)}")
        else:
            col3.metric("모델", metadata.get("model", "Unknown"))


def render_chat_controls() -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ 채팅 기록 삭제", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        export_data = AppService.generate_chat_export_data()
        st.download_button(
            label="💾 채팅 내보내기",
            data=(export_data.encode("utf-8") if export_data else b""),
            file_name=f"chat_history_{int(time.time())}.md",
            mime="text/markdown",
            key="download_chat",
            disabled=not export_data,
        )
        if not export_data:
            st.caption("내보낼 채팅 기록이 없습니다.")
    with col3:
        st.caption(f"총 {len(st.session_state.get('messages', []))}개의 메시지")


def render_feedback_interface() -> None:
    with st.expander("💬 피드백 보내기"):
        st.write("답변의 품질을 평가해주세요:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 도움이 되었어요", key="feedback_positive"):
                st.success("긍정적인 피드백 감사합니다!")
        with col2:
            if st.button("👎 개선이 필요해요", key="feedback_negative"):
                st.info("피드백 감사합니다. 더 나은 답변을 제공하도록 노력하겠습니다!")
        detailed_feedback = st.text_area(
            "상세한 피드백 (선택사항)",
            placeholder="답변에 대한 구체적인 의견이나 개선 사항을 알려주세요...",
            key="detailed_feedback",
        )
        if st.button("피드백 제출", key="submit_feedback") and detailed_feedback:
            st.success("상세한 피드백이 제출되었습니다. 감사합니다!")
