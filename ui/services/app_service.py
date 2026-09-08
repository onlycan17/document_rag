"""애플리케이션 서비스 레이어"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.utils.image_tools import (
    encode_image_to_data_url,
    extract_images_from_documents,
)

logger = logging.getLogger(__name__)

MAX_CHAT_IMAGES = 4


class AppService:
    """애플리케이션 공통 서비스를 제공하는 클래스"""

    @staticmethod
    def serve_image(image_path: str) -> Optional[str]:
        """이미지를 base64 URL로 변환"""
        return encode_image_to_data_url(Path(image_path))

    @staticmethod
    def extract_images_from_content(
        content: str,
        context_documents: List[Any],
        *,
        debug: bool = False,
    ) -> List[Dict[str, str]]:
        """문서 메타데이터에서 이미지 정보 추출"""
        _ = content  # 내용은 현재 사용하지 않지만 시그니처 유지
        images = extract_images_from_documents(
            context_documents,
            max_images=MAX_CHAT_IMAGES,
            debug=debug,
            logger=logger,
        )
        return [info.to_dict() for info in images]

    @staticmethod
    def display_images_in_response(
        content: str,
        context_documents: List[Any],
        *,
        debug: bool = False,
    ) -> str:
        """응답 본문에 이미지 섹션 추가"""
        images = AppService.extract_images_from_content(content, context_documents, debug=debug)
        if not images:
            return content
        image_section = AppService._build_image_section(images)
        return f"{content}\n\n---\n\n**📷 관련 이미지:**\n\n{image_section}"

    @staticmethod
    def format_chat_message_with_metadata(message: Dict[str, Any]) -> str:
        """참고 문서를 포함해 채팅 메시지 포맷"""
        content = message.get("content", "")
        sources = message.get("sources", [])
        if not sources:
            return content
        lines = ["", "**📌 참고 문서:**"]
        lines.extend(AppService._format_source_lines(sources))
        return content + "\n".join(lines)

    @staticmethod
    def extract_source_information(response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """RAG 응답에서 소스 정보 추출"""
        sources: List[Dict[str, Any]] = []
        for doc in response_data.get("context_documents", []):
            metadata = getattr(doc, "metadata", None) or {}
            preview = getattr(doc, "page_content", "")[:200]
            sources.append(
                {
                    "file_name": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A"),
                    "relevance_score": metadata.get("relevance_score", 0),
                    "content_preview": preview,
                }
            )
        return sources

    @staticmethod
    def generate_chat_export_data() -> str:
        """채팅 히스토리를 Markdown으로 변환"""
        messages = st.session_state.get("messages", [])
        if not messages:
            return ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# 채팅 기록 - {timestamp}\n"]
        for index, message in enumerate(messages, start=1):
            lines.extend(AppService._format_chat_entry(index, message))
        return "\n".join(lines)

    @staticmethod
    def validate_file_upload(uploaded_file: Any) -> Dict[str, Any]:
        """업로드 파일 검증"""
        if not uploaded_file:
            return {"valid": False, "error": "파일이 제공되지 않았습니다."}
        size_check = _validate_file_size(uploaded_file.size)
        if size_check is not None:
            return size_check
        extension_check = _validate_extension(uploaded_file.name)
        if extension_check is not None:
            return extension_check
        return {
            "valid": True,
            "error": None,
            "file_info": {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "extension": Path(uploaded_file.name).suffix.lower(),
            },
        }

    @staticmethod
    def get_performance_metrics(
        processing_time: float,
        metadata: Dict[str, Any],
        context_documents: List[Any],
    ) -> Dict[str, Any]:
        return {
            "processing_time": processing_time,
            "document_count": len(context_documents),
            "total_tokens": metadata.get("total_tokens", 0),
            "prompt_tokens": metadata.get("prompt_tokens", 0),
            "completion_tokens": metadata.get("completion_tokens", 0),
            "model": metadata.get("model", "Unknown"),
            "provider": metadata.get("provider", "Unknown"),
        }

    @staticmethod
    def create_debug_info(
        query: str,
        response_data: Dict[str, Any],
        processing_time: float,
    ) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "processing_time": processing_time,
            "response_length": len(response_data.get("answer", "")),
            "context_document_count": len(response_data.get("context_documents", [])),
            "metadata": response_data.get("metadata", {}),
            "search_info": response_data.get("search_info", {}),
            "model_info": {
                "provider": st.session_state.get("current_provider", "Unknown"),
                "model": st.session_state.get("current_model", "Unknown"),
            },
        }

    @staticmethod
    def _build_image_section(images: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for index, image in enumerate(images, start=1):
            data_url = AppService.serve_image(image["path"])
            if not data_url:
                continue
            lines.append(f"{index}. **{_clean_text(image['filename'])}**  ")
            lines.append(f"![{image['filename']}]({data_url})\n")
            lines.append(f"- 출처: {_clean_text(image.get('source', 'Unknown'))}  ")
            description = image.get("description") or ""
            if description:
                lines.append(f"- 설명: {description}  ")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_source_lines(sources: List[Any]) -> List[str]:
        formatted: List[str] = []
        for index, source in enumerate(sources[:3], start=1):
            info = _normalize_source(source)
            line = f"{index}. **{info['file_name']}**"
            if info["relevance_score"]:
                score = 1 - info["relevance_score"]
                line += f" (관련도: {score:.1%})"
            formatted.append(line)
            if info["content_preview"]:
                formatted.append(f"   _{info['content_preview']}..._")
        return formatted

    @staticmethod
    def _format_chat_entry(index: int, message: Dict[str, Any]) -> List[str]:
        role = "사용자" if message.get("role") == "user" else "어시스턴트"
        lines = [f"## {index}. {role}", message.get("content", ""), ""]
        if message.get("role") == "assistant" and "processing_time" in message:
            info = f"*처리 시간: {message['processing_time']:.2f}초, 참조 문서: {len(message.get('context_documents', []))}개*"
            lines.append(info)
            lines.append("")
        lines.append("---")
        lines.append("")
        return lines


def _clean_text(text: str) -> str:
    return text.replace("+", " ") if text else "Unknown"


def _normalize_source(source: Any) -> Dict[str, Any]:
    if isinstance(source, dict):
        return {
            "file_name": source.get("file_name", "Unknown"),
            "relevance_score": source.get("relevance_score", 0),
            "content_preview": source.get("content_preview", ""),
        }
    metadata = getattr(source, "metadata", {}) or {}
    preview = getattr(source, "page_content", "")[:100]
    return {
        "file_name": metadata.get("file_name", "Unknown"),
        "relevance_score": metadata.get("relevance_score", 0),
        "content_preview": preview,
    }


def _validate_file_size(size: int) -> Optional[Dict[str, Any]]:
    max_size = 100 * 1024 * 1024
    if size > max_size:
        error = f"파일 크기가 너무 큽니다. (최대 100MB, 현재: {size / (1024 * 1024):.1f}MB)"
        return {"valid": False, "error": error}
    return None


def _validate_extension(filename: str) -> Optional[Dict[str, Any]]:
    if not filename:
        return {"valid": False, "error": "유효하지 않은 파일명입니다."}
    allowed = {".txt", ".md", ".pdf", ".docx"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        return {"valid": False, "error": f"지원하지 않는 파일 형식입니다. ({ext})"}
    return None
