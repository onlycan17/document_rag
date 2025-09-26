"""문서 추출 파이프라인 헬퍼"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.loaders.document_loader import DocumentLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class DocumentPipelineResult:
    """문서 로딩 결과"""

    documents: List[Any]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return bool(self.documents) and not self.error


def load_documents(
    file_path: str,
    *,
    use_ocr: bool = True,
    use_agent_preprocessing: bool = False,
    enable_postprocessing: bool = True,
) -> DocumentPipelineResult:
    try:
        loader = DocumentLoader(
            use_ocr=use_ocr,
            use_agent_preprocessing=use_agent_preprocessing,
            enable_postprocessing=enable_postprocessing,
        )
        documents = loader.load_document(file_path)
        if not documents:
            return DocumentPipelineResult([], "문서를 추출할 수 없습니다")
        return DocumentPipelineResult(documents)
    except Exception as exc:
        LOGGER.error("문서 로딩 실패: %s", exc)
        return DocumentPipelineResult([], str(exc))


def concatenate_documents(documents: List[Any]) -> str:
    return "\n\n".join(getattr(doc, "page_content", "") for doc in documents)


def build_metadata(
    *,
    original_length: int,
    processed_length: int,
    document_count: int,
    file_path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = {
        "original_length": original_length,
        "processed_length": processed_length,
        "document_count": document_count,
        "file_path": file_path,
    }
    if extra:
        metadata.update(extra)
    return metadata
