"""이미지 경로 처리와 인코딩 유틸"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageInfo:
    """이미지 표시용 정보를 담는 자료 구조"""

    path: str
    filename: str
    source: str
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": self.path,
            "filename": self.filename,
            "source": self.source,
            "description": self.description,
        }


def resolve_image_path(image_ref: str, search_roots: Optional[Sequence[Path]] = None) -> Optional[Path]:
    if not image_ref:
        return None
    candidate = Path(image_ref)
    if candidate.exists():
        return candidate.resolve()
    normalized = _normalize_static_pdf_path(image_ref)
    for path in _iter_candidate_paths(normalized, search_roots):
        if path.exists():
            return path.resolve()
    return None


def encode_image_to_data_url(path: Path) -> Optional[str]:
    if not path or not path.exists():
        return None
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    try:
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{encoded}"
    except Exception as exc:  # pragma: no cover - 파일 접근 실패는 로깅만 수행
        LOGGER.error("이미지 인코딩 실패: %s", exc)
        return None


def extract_images_from_documents(
    context_documents: Iterable[Any],
    *,
    max_images: int = 4,
    debug: bool = False,
    search_roots: Optional[Sequence[Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[ImageInfo]:
    log = logger or LOGGER
    images: List[ImageInfo] = []
    seen: Set[str] = set()
    for doc_index, doc in enumerate(context_documents):
        metadata = _document_metadata(doc)
        for ref, meta in _iter_image_references(metadata):
            resolved = resolve_image_path(ref, search_roots)
            if not resolved:
                if debug:
                    log.debug("이미지 경로 미해결: doc=%s ref=%s", doc_index, ref)
                continue
            info = _build_image_info(resolved, meta)
            if info.path in seen:
                continue
            seen.add(info.path)
            images.append(info)
            if max_images and len(images) >= max_images:
                return images
    return images


def _normalize_static_pdf_path(image_ref: str) -> str:
    if not image_ref.startswith("static/images/pdf/"):
        return image_ref
    filename = os.path.basename(image_ref)
    if "_page" not in filename or "_img" not in filename:
        return filename
    doc_name, page_part = filename.split("_page", 1)
    page_part = page_part.replace("_page", "_p").replace("_img", "_i")
    return f"data/extracted_images/{doc_name}/images/{doc_name}_p{page_part}"


def _iter_candidate_paths(image_ref: str, search_roots: Optional[Sequence[Path]]) -> Iterator[Path]:
    cwd = Path.cwd()
    roots = list(search_roots or _default_search_roots(cwd))
    yield from (Path(image_ref),)
    yield from ((root / image_ref).resolve() for root in roots)
    yield from _search_by_filename(Path(image_ref).name, roots)


def _default_search_roots(cwd: Path) -> List[Path]:
    return [
        cwd / "data" / "extracted_images",
        cwd / "processed_docs",
        cwd / "converted_docs",
        cwd / "data",
        cwd / "static",
        cwd / "static" / "images",
        cwd / "static" / "images" / "pdf",
        cwd / "static" / "images" / "docx",
        cwd / "prompts",
        cwd / "docs",
    ]


def _search_by_filename(name: str, roots: Sequence[Path]) -> Iterator[Path]:
    if not name:
        return
    for root in roots:
        candidate = (root / name).resolve()
        if candidate.exists():
            yield candidate
        name_without_ext = Path(name).stem
        for ext in SUPPORTED_EXTENSIONS:
            maybe = (root / f"{name_without_ext}{ext}").resolve()
            if maybe.exists():
                yield maybe


def _document_metadata(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        return doc.get("metadata", doc)
    return getattr(doc, "metadata", {}) or {}


def _iter_image_references(metadata: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    fields = [
        "image_paths",
        "image_files",
        "images",
        "intelligent_images",
    ]
    for field in fields:
        items = metadata.get(field, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    ref = item.get("image_file") or item.get("image_path") or item.get("path") or item.get("relative_path")
                    if ref:
                        yield ref, {**metadata, **item}
                elif isinstance(item, str):
                    yield item, metadata
    for single_key in ("image_file", "image_path", "relative_path"):
        value = metadata.get(single_key)
        if isinstance(value, str):
            yield value, metadata


def _build_image_info(path: Path, metadata: Dict[str, Any]) -> ImageInfo:
    filename = metadata.get("filename") or path.name
    source = metadata.get("source") or metadata.get("file_name") or "Unknown"
    description = metadata.get("image_description") or metadata.get("description", "")
    return ImageInfo(path=str(path), filename=filename, source=source, description=description)
