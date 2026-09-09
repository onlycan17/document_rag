"""
PDF 로딩 모듈 수준 헬퍼

provider/model 해석, 이미지 스캔·복사, 변환 메타데이터 생성, MD 저장·2단계 후처리 등
PdfLoadingMixin과 독립 검증 가능한 순수/모듈 함수만 모은 모듈이다.
"""

import os
import re
from pathlib import Path
import logging
from datetime import datetime

from config import settings
from ..utils.md_postprocessor import MDPostProcessor
from ..utils.quality_checker import QualityChecker

logger = logging.getLogger(__name__)


def resolve_provider_model(provider: str | None, model: str | None) -> tuple[str, str | None]:
    """provider별 설정 기본 모델을 완성한다 (MD 후처리·에이전트 변환 공통)."""
    provider = (provider or getattr(settings, "llm_provider", "openrouter")).lower()
    if model:
        return provider, model
    per_provider = {"openai": "openai_model", "google": "google_model", "anthropic": "anthropic_model"}
    if provider in per_provider:
        return provider, getattr(settings, per_provider[provider], None)
    if provider == "openrouter":
        # 텍스트용 openrouter_model 우선, 없으면 멀티모달 기본 사용
        return provider, getattr(settings, "openrouter_model", None) or getattr(settings, "openrouter_mm_model", None)
    return provider, None


def scan_images_dir(temp_output_dir: str | Path) -> list[dict]:
    """임시 변환 디렉토리의 images/에서 추출 이미지 정보를 수집한다."""
    extracted_images: list[dict] = []
    images_dir = Path(temp_output_dir) / "images"
    if not images_dir.exists():
        return extracted_images

    # 파일명 정규화가 적용되도록 안전화된 스템으로 매칭 폭을 넓힘
    for img_path in images_dir.glob("*_page*_img*.png"):
        try:
            stat = img_path.stat()
            page_num = None
            m = re.search(r"_page(\d+)_img(\d+)", img_path.name)
            if m:
                page_num = int(m.group(1))
            extracted_images.append(
                {
                    "filename": img_path.name,
                    "path": str(img_path),
                    "relative_path": f"converted_docs/images/{img_path.name}",
                    "size": stat.st_size,
                    "format": "PNG",
                    "content_type": "image/png",
                    "extracted_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "page": page_num,
                    "description": f"페이지 {page_num} 이미지" if page_num else "추출된 이미지",
                }
            )
        except Exception as e:
            logger.warning(f"PDF 이미지 정보 수집 실패: {str(e)}")

    if extracted_images:
        logger.info(f"   📁 temp_output_dir에서 {len(extracted_images)}개 이미지 발견")
    return extracted_images


def copy_images_to_permanent(extracted_images: list[dict], dest_dir: Path = Path("static/images/pdf")) -> None:
    """추출 이미지를 영구 위치로 복사하고 각 정보의 경로를 갱신한다."""
    import shutil

    dest_dir.mkdir(parents=True, exist_ok=True)
    for image_info in extracted_images:
        dst_path = dest_dir / image_info["filename"]
        try:
            shutil.copy2(image_info["path"], dst_path)
            image_info["path"] = str(dst_path)
            image_info["relative_path"] = f"static/images/pdf/{image_info['filename']}"
            logger.info(f"   🖼️  PDF 이미지 이동: {image_info['filename']}")
        except Exception as e:
            logger.warning(f"PDF 이미지 이동 실패: {str(e)}")


def build_extraction_metadata(extraction_results: dict, file_path: str | Path, output_dir: Path) -> dict:
    """지능형 이미지 추출 결과를 메타데이터+보고서 마크다운으로 변환한다."""
    saved_images = []
    extracted_texts = []

    for img in extraction_results["images"]:
        if img.get("saved"):
            saved_images.append(
                {
                    "filename": Path(img["image_file"]).name,
                    "path": img["image_file"],
                    "page": img["page"],
                    "type": img.get("type", "unknown"),
                    "relevance_score": img.get("relevance_score", 0),
                    "description": img.get("description", ""),
                }
            )
        if img.get("extracted_text"):
            extracted_texts.append(img["extracted_text"])

    markdown_content = f"# {Path(file_path).name}\n\n"
    markdown_content += (
        f"**문서 주제**: {extraction_results.get('document_topic', {}).get('main_topic', '알 수 없음')}\n\n"
    )

    if saved_images:
        markdown_content += "## 추출된 이미지\n\n"
        for img_info in saved_images:
            markdown_content += (
                f"- 페이지 {img_info['page']}: {img_info['filename']} " f"(관련도: {img_info['relevance_score']:.2f})\n"
            )
        markdown_content += "\n"

    if extracted_texts:
        markdown_content += "## OCR 추출 텍스트\n\n"
        markdown_content += "\n\n".join(extracted_texts)

    stats = extraction_results.get("statistics", {})
    return {
        "intelligent_extraction_completed": True,
        "document_topic": extraction_results.get("document_topic", {}),
        "total_images": stats.get("total_images_found", 0),
        "relevant_images": stats.get("relevant_images_saved", 0),
        "text_images_converted": stats.get("text_images_converted", 0),
        "extracted_images": saved_images,
        "image_extraction_dir": str(output_dir),
        "image_markdown_content": markdown_content,
    }


def convert_agent_images_to_metadata(images_info: dict, temp_output_dir: str) -> dict:
    """에이전트 변환기 이미지 정보를 지능형 추출 메타데이터 형식으로 변환한다."""
    extracted_images = []
    for page_num, images in images_info.items():
        for img_path, img_description in images:
            if not os.path.isabs(img_path):
                img_path = os.path.join(temp_output_dir, img_path)
            extracted_images.append(
                {
                    "filename": os.path.basename(img_path),
                    "path": img_path,
                    "page": page_num,
                    "description": img_description,
                    "source": "agent_converter",
                }
            )

    return {
        "extracted_images": extracted_images,
        "total_images": len(extracted_images),
        "relevant_images": len(extracted_images),
        "text_images_converted": 0,
        "image_extraction_dir": os.path.join(temp_output_dir, "images"),
        "document_topic": {"main_topic": "PDF 문서", "keywords": []},
    }


def cleanup_temp_dir(path: str) -> None:
    """임시 변환 디렉토리를 조용히 정리한다."""
    import shutil

    try:
        shutil.rmtree(path)
    except Exception as err:
        logger.debug(f"임시 출력 디렉터리 정리 실패(무시): {err}")


def write_conversion_md(file_path: str | Path, markdown_content: str, method_name: str, image_count: int) -> Path:
    """변환 결과 MD 파일을 전처리 확인용으로 저장한다."""
    md_dir = Path("converted_docs")
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(file_path).stem
    md_file_path = md_dir / f"{pdf_name}.md"
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(f"# {pdf_name}\n\n")
        f.write(f"**원본 파일**: {Path(file_path).name}\n")
        f.write(f"**변환 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**처리 방법**: {method_name}\n")
        f.write(f"**추출된 이미지**: {image_count}개\n\n")
        f.write("---\n\n")
        f.write(markdown_content)
    return md_file_path


def resolve_postprocess_target() -> tuple[str, str | None]:
    """MD 후처리 provider/model을 세션→env 오버라이드→설정 순으로 해석한다."""
    provider = model = None
    try:
        import streamlit as st  # type: ignore

        provider = st.session_state.get("current_provider", None)
        model = st.session_state.get("current_model", None)
    except Exception as err:
        logger.debug(f"세션 provider/model 조회 실패(무시): {err}")

    override_provider = getattr(settings, "md_postprocess_provider", None)
    override_model = getattr(settings, "md_postprocess_model", None)
    if override_provider:
        provider = override_provider
        if override_model:
            model = override_model

    return resolve_provider_model(provider, model)


def run_md_postprocessing(md_file_path: Path, document) -> None:
    """2단계 MD 후처리를 실행하고 결과로 document를 갱신한다."""
    try:
        logger.info("🔧 2단계 MD 후처리 시작...")

        provider, model = resolve_postprocess_target()
        postprocessor = MDPostProcessor(
            output_dir="processed_docs",
            target_quality=settings.md_postprocess_target_quality,
            provider=provider,
            model_name=model,
        )
        postprocessor.set_quality_checker(QualityChecker(provider=provider, model_name=model))

        processed_path = postprocessor.process_file(str(md_file_path))
        if not processed_path:
            logger.warning("2단계 후처리 실패 - 원본 유지")
            document.metadata["postprocessed"] = False
            return

        logger.info(f"   ✅ 2단계 후처리 완료: {processed_path}")
        document.page_content = Path(processed_path).read_text(encoding="utf-8")
        document.metadata["postprocessed"] = True
        document.metadata["processed_md_path"] = str(processed_path)
        document.metadata["processing_quality"] = postprocessor.last_quality_score

    except Exception as e:
        logger.error(f"2단계 후처리 중 오류: {str(e)}")
        document.metadata["postprocessed"] = False
