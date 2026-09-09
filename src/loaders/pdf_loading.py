"""PDF 파일 로딩·변환 결과 처리 전용 믹스인"""

from typing import Dict, List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import os
from pathlib import Path
from config import settings
from ..utils.pdf_converter import ImprovedPDFConverter
from .pdf_loading_helpers import (  # noqa: F401 - 외부 import 경로 유지
    build_extraction_metadata,
    cleanup_temp_dir,
    convert_agent_images_to_metadata,
    copy_images_to_permanent,
    resolve_postprocess_target,
    resolve_provider_model,
    run_md_postprocessing,
    scan_images_dir,
    write_conversion_md,
)
from ..utils.agent_pdf_converter import AgentBasedPDFConverter
from ..utils.text_processing import TextProcessor
import logging

logger = logging.getLogger(__name__)


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
        extracted_images = self._collect_extracted_images(temp_output_dir)

        base_metadata = self._build_pdf_metadata(file_path, processing_method, image_count, extracted_images)
        markdown_content = self._prepend_intelligent_image_content(base_metadata, markdown_content)

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

        self._save_md_and_postprocess(document, file_path, markdown_content, extracted_images, method_name)

        return [document]

    def _collect_extracted_images(self, temp_output_dir: str) -> list:
        """지능형 추출 정보를 우선하고 없으면 임시 디렉토리를 스캔, 영구 위치로 복사"""
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

        return extracted_images

    def _build_pdf_metadata(
        self, file_path: str, processing_method: str, image_count: int, extracted_images: list
    ) -> Dict:
        """PDF 변환 Document의 기본 메타데이터 생성(지능형 추출 정보 병합 포함)"""
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

        return base_metadata

    def _prepend_intelligent_image_content(self, base_metadata: Dict, markdown_content: str) -> str:
        """이미지 추출 내용을 PDF 텍스트 내용 앞에 추가"""
        if not (hasattr(self, "image_extraction_metadata") and self.image_extraction_metadata):
            return markdown_content

        image_content = self.image_extraction_metadata.get("image_markdown_content", "")
        if image_content:
            markdown_content = f"{image_content}\n\n---\n\n# PDF 텍스트 내용\n\n{markdown_content}"
        return markdown_content

    def _save_md_and_postprocess(
        self,
        document: Document,
        file_path: str,
        markdown_content: str,
        extracted_images: list,
        method_name: str,
    ) -> None:
        """MD 파일 저장(전처리 확인용) 및 2단계 후처리"""
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
