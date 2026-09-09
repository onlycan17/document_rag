"""PDF 파일 로딩·변환 결과 처리 전용 믹스인"""

from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import os
import re
from pathlib import Path
from config import settings
from ..utils.pdf_converter import ImprovedPDFConverter
from ..utils.agent_pdf_converter import AgentBasedPDFConverter
from ..utils.md_postprocessor import MDPostProcessor
from ..utils.quality_checker import QualityChecker
from ..utils.text_processing import TextProcessor
import logging
from datetime import datetime

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


class PdfLoadingMixin:
    """PDF 파일 로딩·변환 결과 처리 전용 믹스인 — EnhancedDocumentLoader 믹스인"""

    def _process_converted_content(
        self,
        markdown_content: str,
        file_path: str,
        temp_output_dir: str,
        progress_callback=None,
        processing_method: str = "pdf_converter",
        image_count: int = 0,
    ) -> List[Document]:
        """변환된 마크다운 콘텐츠와 이미지를 처리하여 Document 객체 생성"""

        # PDF에서 추출된 이미지 정보 수집
        extracted_images = []

        # 우선순위 1: 지능형 이미지 추출 정보가 있으면 우선 사용
        if hasattr(self, "image_extraction_metadata") and self.image_extraction_metadata:
            intelligent_images = self.image_extraction_metadata.get("extracted_images", []) or []
            if intelligent_images:
                logger.info(f"   🎯 지능형 이미지 추출 정보 발견: {len(intelligent_images)}개 이미지")
                extracted_images = intelligent_images

        # 우선순위 2: temp_output_dir의 이미지 디렉토리에서 찾기
        if not extracted_images:
            extracted_images = scan_images_dir(temp_output_dir)

        # 이미지를 영구 위치로 복사
        if extracted_images:
            copy_images_to_permanent(extracted_images)

        # 기본 메타데이터 생성
        base_metadata = {
            "source": str(file_path),
            "file_name": Path(file_path).name,
            "file_type": ".pdf",
            "processing_method": processing_method,
            "image_count": image_count or len(extracted_images),
            "images": extracted_images,
            "conversion_status": "success",
        }

        # 지능형 이미지 추출 정보가 있으면 추가 메타데이터 병합
        if hasattr(self, "image_extraction_metadata") and self.image_extraction_metadata:
            # 지능형 추출 정보 병합 (images는 이미 extracted_images에 포함되어 있음)
            base_metadata.update(
                {
                    "intelligent_extraction_completed": True,
                    "document_topic": self.image_extraction_metadata.get("document_topic", {}),
                    "total_images": self.image_extraction_metadata.get("total_images", 0),
                    "relevant_images": self.image_extraction_metadata.get("relevant_images", 0),
                    "text_images_converted": self.image_extraction_metadata.get("text_images_converted", 0),
                    "image_extraction_dir": self.image_extraction_metadata.get("image_extraction_dir", ""),
                }
            )
            logger.info(
                f"   ✅ 지능형 이미지 추출 메타데이터 병합 완료 (관련 이미지: {base_metadata.get('relevant_images', 0)}개)"
            )

            # 이미지 추출 내용을 PDF 텍스트 내용 앞에 추가
            image_content = self.image_extraction_metadata.get("image_markdown_content", "")
            if image_content:
                markdown_content = f"{image_content}\n\n---\n\n# PDF 텍스트 내용\n\n{markdown_content}"

        # 마크다운 내용을 Document 객체로 변환
        document = Document(page_content=markdown_content, metadata=base_metadata)

        method_name = "에이전트 기반 변환기" if "agent" in processing_method else "개선된 PDF 변환기"

        # 이미지 메타데이터 검증 로그
        total_images = len(base_metadata.get("images", []))
        logger.info(f"{method_name}로 처리 완료: {file_path}")
        logger.info(
            f"   🖼️  Document 메타데이터: images 필드 포함={('images' in base_metadata)}, 총 {total_images}개 이미지"
        )
        if total_images > 0:
            logger.info("   ✅ 이미지 메타데이터가 Document 객체에 정상적으로 포함됨")

        # MD 파일 저장 (전처리 확인용) 및 2단계 후처리
        try:
            md_file_path = write_conversion_md(file_path, markdown_content, method_name, len(extracted_images))

            document.metadata["md_file_path"] = str(md_file_path)
            document.metadata["md_saved"] = True

            logger.info(f"   📝 MD 파일 저장: {md_file_path}")

            if settings.enable_md_postprocessing:
                run_md_postprocessing(md_file_path, document)
            else:
                logger.info("2단계 후처리 비활성화됨")
                document.metadata["postprocessed"] = False

        except Exception as e:
            logger.warning(f"MD 파일 저장 실패: {str(e)}")
            document.metadata["md_saved"] = False

        return [document]

    def _load_pdf_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """PDF 파일 로딩 - 에이전트 모드, 지능형 이미지 추출, 또는 개선된 PDF 변환기 사용"""

        # 중복 업로드 대비: 동일 문서의 기존 이미지 정리
        try:
            self._cleanup_existing_images_for_pdf(file_path)
        except Exception as _cleanup_err:
            logger.warning(f"기존 이미지 정리 중 경고: {str(_cleanup_err)}")

        # 각 파일 처리 시작 시 메타데이터 초기화
        self.image_extraction_metadata = None

        if self.use_intelligent_image_extraction:
            self._extract_intelligent_images(file_path, progress_callback)

        if self.use_agent_preprocessing:
            documents = self._load_with_agent_converter(file_path, progress_callback)
            if documents is not None:
                return documents

        documents = self._load_with_improved_converter(file_path, progress_callback)
        if documents is not None:
            return documents

        documents = self._load_with_ocr(file_path, progress_callback)
        if documents is not None:
            return documents

        return self._load_with_basic_loader(file_path, progress_callback)

    def _extract_intelligent_images(self, file_path: str, progress_callback) -> None:
        """지능형 이미지 추출(OpenRouter 전용, 정책 강제) 수행 후 메타데이터 저장"""
        try:
            if progress_callback:
                progress_callback(0.1, "🧠 지능형 이미지 추출 중...")
            else:
                logger.info("🧠 지능형 이미지 추출 시작 (provider=openrouter, 정책상 강제)")
            from pathlib import Path

            # PDF 파일명 기반으로 출력 디렉토리 생성
            pdf_name = Path(file_path).stem
            output_base_dir = Path("data/extracted_images")
            output_dir = output_base_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # OpenRouter만 사용(정책 강제), 폴백 없음
            from ..utils.openrouter_image_service import OpenRouterImageService

            try:
                svc = OpenRouterImageService()
                logger.info("OpenRouter 기반 지능형 이미지 추출 경로 선택(정책 강제)")
                extraction_results = svc.process_pdf(
                    pdf_path=file_path,
                    output_dir=str(output_dir),
                    relevance_threshold=settings.local_image_relevance_threshold,
                )
                logger.info("OpenRouter를 이용한 지능형 추출 완료")
            except Exception as e:
                # 정책상 폴백 금지: 즉시 중단
                raise RuntimeError(f"OpenRouter 이미지 분석 실패(폴백 금지 정책): {e}") from e

            # 추출된 텍스트와 관련 이미지 정보를 Document로 변환
            if extraction_results and extraction_results.get("images"):
                image_extraction_metadata = build_extraction_metadata(extraction_results, file_path, output_dir)
                saved_count = len(image_extraction_metadata["extracted_images"])

                # 추출 보고서 로그
                logger.info(f"✅ 지능형 이미지 추출 완료: {file_path}")
                logger.info(f"   📁 이미지 저장 위치: {output_dir}")
                logger.info(f"   🖼️  관련 이미지: {saved_count}개 저장됨")

                if progress_callback:
                    progress_callback(
                        0.3, f"지능형 이미지 추출 완료! (관련 이미지 {saved_count}개), PDF 텍스트 처리 중..."
                    )

                # 이미지 추출 정보를 저장하고 텍스트 처리 계속
                self.image_extraction_metadata = image_extraction_metadata

            logger.info("지능형 이미지 추출 완료, 기존 방식으로 텍스트 추출 진행")

        except ImportError as e:
            logger.warning(f"이미지 추출 서비스를 사용할 수 없습니다: {str(e)}")
        except Exception as e:
            logger.warning(f"지능형 이미지 추출 실패, 기존 방식으로 폴백: {str(e)}")

    def _resolve_agent_provider_model(self) -> tuple:
        """에이전트 LLM 제공자/모델을 UI 선택값 또는 설정으로 강제 동기화"""
        provider = None
        model = None
        try:
            import streamlit as st  # type: ignore

            # 전처리 섹션의 선택값을 최우선으로 사용
            provider = st.session_state.get("preprocessing_model", None) or self.preprocessing_model
            if st.session_state.get("enable_multimodal_preprocessing", False):
                model = st.session_state.get("preproc_mm_model", None)
            else:
                model = st.session_state.get("preproc_text_model", None)
        except Exception:
            # 세션을 사용할 수 없으면 인자로 받은 전처리 모델 타입 사용
            provider = self.preprocessing_model

        return resolve_provider_model(provider, model)

    def _load_with_agent_converter(self, file_path: str, progress_callback) -> Optional[List[Document]]:
        """에이전트 기반 PDF 변환. 실패 시 None 반환(상위 폴백 유도)"""
        try:
            if progress_callback:
                progress_callback(0.1, "🤖 에이전트 기반 고품질 변환 중...")

            # 임시 출력 디렉토리 사용
            import tempfile

            temp_output_dir = tempfile.mkdtemp(prefix="agent_pdf_convert_")
            provider, model = self._resolve_agent_provider_model()

            agent_converter = AgentBasedPDFConverter(
                output_dir=temp_output_dir,
                enable_quality_validation=True,
                llm_provider=provider,
                llm_model=model,
            )

            # 에이전트 기반 PDF 변환
            markdown_path = agent_converter.convert_pdf_to_markdown(file_path)

            if markdown_path and os.path.exists(markdown_path):
                with open(markdown_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

                # 에이전트 변환기에서 추출한 이미지 메타데이터 가져오기
                if hasattr(agent_converter, "extracted_images_info") and agent_converter.extracted_images_info:
                    logger.info("🖼️  에이전트 변환기에서 추출된 이미지 정보 발견")

                    self.image_extraction_metadata = convert_agent_images_to_metadata(
                        agent_converter.extracted_images_info, temp_output_dir
                    )
                    logger.info(
                        f"   ✅ 에이전트 이미지 메타데이터 변환 완료: {self.image_extraction_metadata['total_images']}개"
                    )

                # 기존 이미지 처리 로직 재사용
                return self._process_converted_content(
                    markdown_content,
                    file_path,
                    temp_output_dir,
                    progress_callback,
                    processing_method="agent_based_converter",
                )

            return None
        except Exception as e:
            logger.warning(f"에이전트 변환 실패, 기존 방식으로 폴백: {str(e)}")
            # 에이전트 실패 시 기존 방식으로 폴백
            return None

    def _load_with_improved_converter(self, file_path: str, progress_callback) -> Optional[List[Document]]:
        """개선된 PDF 변환기(문장 연결성 향상) 로딩. 실패 시 None 반환(OCR 폴백 유도)"""
        temp_output_dir = None
        try:
            if progress_callback:
                progress_callback(0.1, "개선된 PDF 변환기로 처리 중...")
            else:
                logger.info("📝 개선된 PDF 변환기 시작: 텍스트/이미지 추출 수행")

            # 임시 출력 디렉토리 사용
            import tempfile

            temp_output_dir = tempfile.mkdtemp(prefix="pdf_convert_")
            pdf_converter = ImprovedPDFConverter(
                output_dir=temp_output_dir, enable_postprocessing=self.enable_postprocessing
            )

            # PDF를 마크다운으로 변환
            markdown_content, image_count = pdf_converter.convert_pdf_to_markdown(file_path, progress_callback)

            if markdown_content:
                # 공통 이미지 처리 로직 사용
                return self._process_converted_content(
                    markdown_content,
                    file_path,
                    temp_output_dir,
                    progress_callback,
                    processing_method="improved_pdf_converter_with_images",
                    image_count=image_count,
                )

            logger.warning(f"개선된 PDF 변환기에서 내용 추출 실패: {file_path}")
            # 임시 디렉토리 정리
            cleanup_temp_dir(temp_output_dir)
            return None

        except Exception as e:
            logger.warning(f"개선된 PDF 변환기 실패, OCR 모드로 전환: {str(e)}")
            # 임시 디렉토리 정리
            if temp_output_dir:
                import shutil

                try:
                    shutil.rmtree(temp_output_dir)
                except Exception as err:
                    logger.debug(f"임시 출력 디렉터리 정리 실패(무시): {err}")
            if progress_callback:
                progress_callback(0.3, "OCR 모드로 전환 중...")
            return None

    def _load_with_ocr(self, file_path: str, progress_callback) -> Optional[List[Document]]:
        """2차 시도: OCR을 사용한 고급 PDF 로더. 미활성화/실패 시 None 반환"""
        if not self.use_ocr:
            return None
        try:
            if progress_callback:
                progress_callback(0.4, "PDF 분석 중... (OCR 모드)")
            documents = self.advanced_pdf_loader.load_pdf(file_path, progress_callback)
            logger.info(f"고급 PDF 로더로 처리: {file_path}")
            return documents
        except Exception as e:
            logger.warning(f"고급 PDF 로더 실패, 기본 로더 사용: {str(e)}")
            if progress_callback:
                progress_callback(0.6, "기본 PDF 로더로 전환...")
            return None

    def _load_with_basic_loader(self, file_path: str, progress_callback) -> List[Document]:
        """3차 시도: 기본 PDF 로더(최후 수단)"""
        if progress_callback:
            progress_callback(0.7, "기본 PDF 텍스트 추출 중...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if progress_callback:
            progress_callback(0.9, "기본 텍스트 추출 완료")

        # 기본 로더 사용 시 메타데이터 보강
        for doc in documents:
            doc.metadata.update({"processing_method": "basic_pdf_loader", "conversion_status": "fallback"})

        return documents

    def _cleanup_existing_images_for_pdf(self, file_path: str) -> None:
        """동일 PDF 문서 재업로드 시 기존 이미지를 정리"""
        try:
            from pathlib import Path
            import shutil

            pdf_name = Path(file_path).stem
            # 파일명 정규화 규칙에 맞춘 프리픽스 생성
            safe_stem = TextProcessor.sanitize_filename(pdf_name)

            # 1) 정적 이미지 저장소(static/images/pdf) 정리
            pdf_images_dir = Path("static/images/pdf")
            removed_pdf_images = 0
            if pdf_images_dir.exists():
                for img_path in pdf_images_dir.glob(f"{safe_stem}_page*_img*.*"):
                    try:
                        img_path.unlink()
                        removed_pdf_images += 1
                    except Exception as err:
                        logger.debug(f"PDF 이미지 파일 삭제 실패(무시): {err}")
            if removed_pdf_images > 0:
                logger.info(f"🧹 기존 PDF 이미지 정리: {removed_pdf_images}개 삭제 (static/images/pdf)")

            # 2) 지능형 이미지 추출 디렉토리 정리 (data/extracted_images/<pdf_name>)
            extracted_dir = Path("data/extracted_images") / pdf_name
            if extracted_dir.exists():
                try:
                    shutil.rmtree(extracted_dir)
                    logger.info(f"🧹 기존 지능형 추출 디렉토리 삭제: {extracted_dir}")
                except Exception as e:
                    logger.warning(f"지능형 추출 디렉토리 삭제 실패: {e}")

        except Exception as e:
            # 치명적이지 않으므로 경고로만 기록
            logger.warning(f"이미지 정리 중 예외 발생: {str(e)}")
