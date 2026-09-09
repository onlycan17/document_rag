from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import os
import re
from pathlib import Path
from config import settings
from .pdf_loader_advanced import AdvancedPDFLoader
from ..utils.pdf_converter import ImprovedPDFConverter
from ..utils.agent_pdf_converter import AgentBasedPDFConverter
from ..utils.md_postprocessor import MDPostProcessor
from ..utils.quality_checker import QualityChecker
from ..utils.text_processing import TextProcessor
import logging
from datetime import datetime

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedDocumentLoader:
    """
    향상된 문서 로더 클래스
    - 의미 기반 청킹 지원
    - 한국어 문서 최적화
    - 마크다운 특화 처리 (CLI 스크립트와 동일한 로직)
    - 다양한 청킹 전략 지원
    - 지능형 이미지 추출 지원
    """

    def __init__(
        self,
        use_ocr: bool = True,
        use_agent_preprocessing: bool = False,
        enable_postprocessing: bool = False,
        use_intelligent_image_extraction: bool = False,
        preprocessing_model: str = "openrouter",
        enable_multimodal_preprocessing: bool = False,
    ):
        self.use_ocr = use_ocr
        self.use_agent_preprocessing = use_agent_preprocessing
        self.enable_postprocessing = enable_postprocessing
        self.use_intelligent_image_extraction = use_intelligent_image_extraction
        self.preprocessing_model = preprocessing_model
        self.enable_multimodal_preprocessing = enable_multimodal_preprocessing

        # 지능형 이미지 추출 메타데이터 저장용
        self.image_extraction_metadata = None

        # 전처리 모델 초기화
        self._preprocessing_model = None
        self._initialize_preprocessing_model()

        # 기본 텍스트 분할기 (기존 방식)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "。", "!", "?", ";", "；", ",", "，", " ", ""],
        )

        # 마크다운 전용 분할기 (새로 추가)
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # 마크다운 구조를 고려한 분리자 순서
            separators=[
                "\n\n\n",  # 여러 줄바꿈 (섹션 구분)
                "\n\n",  # 문단 분리
                "\n#",  # 제목 구분
                "\n##",  # 하위 제목 구분
                "\n###",  # 세부 제목 구분
                "\n- ",  # 목록 항목
                "\n* ",  # 목록 항목 (별표)
                "\n",  # 일반 줄바꿈
                "。",  # 한국어 마침표
                ".",  # 영어 마침표
                "!",  # 느낌표
                "?",  # 물음표
                ";",  # 세미콜론
                ",",  # 쉼표
                " ",  # 공백
                "",  # 문자 단위
            ],
        )

        # 의미 기반 분할기 (향상된 방식)
        if settings.use_semantic_chunking:
            try:
                self.semantic_splitter = SentenceTransformersTokenTextSplitter(
                    chunk_overlap=settings.chunk_overlap,
                    model_name=getattr(settings, "korean_embedding_model", settings.embedding_model_name),
                    tokens_per_chunk=settings.chunk_size,
                )
                logger.info("의미 기반 청킹 활성화됨")
            except Exception as e:
                logger.warning(f"의미 기반 청킹 초기화 실패, 기본 청킹 사용: {str(e)}")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None

        self.use_ocr = use_ocr
        self.advanced_pdf_loader = AdvancedPDFLoader(use_ocr=use_ocr)

        # OCR 사용 가능 여부 확인
        if use_ocr:
            ocr_available = AdvancedPDFLoader.check_ocr_availability()
            if not ocr_available:
                logger.warning("OCR을 사용할 수 없습니다. Tesseract와 한국어 언어팩을 설치해주세요.")
                self.use_ocr = False

    def _initialize_preprocessing_model(self):
        """전처리 모델을 초기화합니다."""
        try:
            from src.processing.preprocessing_factory import PreprocessingModelFactory

            # 선택된 전처리 모델로 초기화
            # 멀티모달 활성화 시 UI에서 선택된 모델명을 우선 적용,
            # 아니면 텍스트 전처리용 선택 모델(preproc_text_model) 사용
            selected_model_name = None
            try:
                import streamlit as st  # type: ignore

                if st.session_state.get("enable_multimodal_preprocessing", False):
                    selected_model_name = st.session_state.get("preproc_mm_model", None)
                else:
                    selected_model_name = st.session_state.get("preproc_text_model", None)
            except Exception as err:
                logger.debug(f"세션 전처리 모델 이름 조회 실패(무시): {err}")

            self._preprocessing_model = PreprocessingModelFactory.create_model(
                self.preprocessing_model, model_name=selected_model_name
            )
            logger.info(f"전처리 모델 초기화 완료: {self.preprocessing_model}")

        except Exception as e:
            logger.error(f"전처리 모델 초기화 실패: {e}")
            # 실패 시 OpenRouter(외부 API)로 폴백
            try:
                self._preprocessing_model = PreprocessingModelFactory.create_model("openrouter")
                logger.warning("전처리 모델 초기화 실패, OpenRouter로 폴백")
            except Exception as fallback_error:
                logger.error(f"OpenRouter 폴백도 실패: {fallback_error}")
                self._preprocessing_model = None

    def load_document(self, file_path: str, progress_callback=None) -> List[Document]:
        """
        단일 문서를 로드하고 최적화된 청크로 분할
        - 파일 유형별 최적화된 로딩
        - 마크다운 특화 처리
        - 향상된 전처리
        - 의미 기반 청킹 지원
        """
        import time

        start_time = time.time()

        file_extension = Path(file_path).suffix.lower()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_name = Path(file_path).name

        logger.info(f"📄 문서 로딩 시작: {file_name} ({file_size_mb:.1f}MB, {file_extension})")

        if progress_callback:
            progress_callback(0.1, f"파일 로드 중... ({file_size_mb:.1f}MB)")

        try:
            # 1. 파일 유형별 로딩
            load_start = time.time()
            if file_extension == ".md":
                documents = self._load_markdown_file(file_path, progress_callback)
                load_method = "마크다운 전용 로더"
            elif file_extension == ".txt":
                documents = self._load_text_file(file_path, progress_callback)
                load_method = "텍스트 로더"
            elif file_extension == ".pdf":
                documents = self._load_pdf_file(file_path, progress_callback)
                load_method = "PDF 로더"
            elif file_extension == ".docx":
                documents = self._load_docx_file(file_path, progress_callback)
                load_method = "DOCX 로더"
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")

            load_time = time.time() - load_start
            logger.info(f"   ✓ {load_method} 완료: {len(documents)}개 페이지 ({load_time:.1f}초)")

            if progress_callback:
                progress_callback(0.6, f"문서 전처리 중... ({len(documents)}개 페이지)")

            # 2. 문서 전처리 및 정제
            preprocess_start = time.time()
            processed_documents = self._preprocess_documents(documents, file_extension)
            preprocess_time = time.time() - preprocess_start
            logger.info(f"   ✓ 전처리 완료: {len(processed_documents)}개 유효 페이지 ({preprocess_time:.1f}초)")

            if progress_callback:
                progress_callback(0.7, "문서 분할 중... (전처리 완료)")

            # 3. 청킹 전략에 따른 분할
            chunk_start = time.time()
            chunks = self._split_documents_optimized(processed_documents, file_extension)
            chunk_time = time.time() - chunk_start
            logger.info(f"   ✓ 청킹 완료: {len(chunks)}개 청크 ({chunk_time:.1f}초)")

            # 4. 메타데이터 보강
            metadata_start = time.time()
            enhanced_chunks = self._enhance_metadata(chunks, file_path)
            metadata_time = time.time() - metadata_start
            logger.info(f"   ✓ 메타데이터 보강 완료 ({metadata_time:.1f}초)")

            if progress_callback:
                progress_callback(0.9, f"분할 완료! ({len(enhanced_chunks)}개 청크)")

            total_time = time.time() - start_time
            avg_chunk_size = (
                sum(len(chunk.page_content) for chunk in enhanced_chunks) / len(enhanced_chunks)
                if enhanced_chunks
                else 0
            )

            logger.info(f"✅ 문서 로딩 성공: {file_name}")
            logger.info(f"   📊 총 처리 시간: {total_time:.1f}초")
            logger.info(f"   📚 최종 청크 수: {len(enhanced_chunks)}개")
            logger.info(f"   📏 평균 청크 크기: {avg_chunk_size:.0f}자")

            return enhanced_chunks

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ 문서 로딩 실패: {file_name}")
            logger.error(f"   🕒 실패까지 소요 시간: {total_time:.1f}초")
            logger.error(f"   💥 오류 내용: {str(e)}")
            logger.error(f"   📁 파일 경로: {file_path}")
            logger.error(f"   📊 파일 크기: {file_size_mb:.1f}MB")
            raise

    def _load_markdown_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """
        마크다운 파일 전용 로딩 (rebuild_markdown_vector_db.py와 동일한 로직)
        """
        try:
            file_name = Path(file_path).name

            # 다양한 인코딩으로 시도
            encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
            content = None
            used_encoding = None

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"   인코딩 {encoding} 시도 실패: {str(e)}")
                    continue

            if content is None:
                raise ValueError("마크다운 파일 로딩 실패: 지원되는 인코딩 없음")

            logger.info(f"   📖 마크다운 로딩 완료: {used_encoding} 인코딩")

            if progress_callback:
                progress_callback(0.5, "마크다운 텍스트 추출 완료")

            # Document 객체 생성 (마크다운 특화 메타데이터 포함)
            document = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_name": file_name,
                    "file_type": ".md",
                    "encoding": used_encoding,
                    "original_size": len(content),
                    "processing_method": "markdown_optimized",
                },
            )

            return [document]

        except Exception as e:
            logger.error(f"마크다운 파일 로딩 실패: {str(e)}")
            # 기본 텍스트 로더로 폴백
            return self._load_text_file(file_path, progress_callback)

    def _load_text_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """텍스트 파일 로딩 최적화"""
        try:
            # 다양한 인코딩 시도
            encodings = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"파일 인코딩 감지: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise ValueError("지원되는 인코딩을 찾을 수 없습니다")

            if progress_callback:
                progress_callback(0.5, "텍스트 추출 완료")

            return [Document(page_content=content, metadata={"source": file_path})]

        except Exception as e:
            logger.error(f"텍스트 파일 로딩 실패: {str(e)}")
            # 기본 로더로 폴백
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()

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
        Path(file_path).stem
        extracted_images = []

        # 우선순위 1: 지능형 이미지 추출 정보가 있으면 우선 사용
        if hasattr(self, "image_extraction_metadata") and self.image_extraction_metadata:
            intelligent_images = self.image_extraction_metadata.get("extracted_images", []) or []
            if intelligent_images:
                logger.info(f"   🎯 지능형 이미지 추출 정보 발견: {len(intelligent_images)}개 이미지")
                extracted_images = intelligent_images

        # 우선순위 2: temp_output_dir의 이미지 디렉토리에서 찾기
        if not extracted_images:
            images_dir = Path(temp_output_dir) / "images"
            if images_dir.exists():
                # 파일명 정규화가 적용되도록 안전화된 스템으로 매칭 폭을 넓힘
                for img_path in images_dir.glob("*_page*_img*.png"):
                    try:
                        stat = img_path.stat()
                        # 페이지/인덱스 파싱
                        page_num = None
                        try:
                            m = re.search(r"_page(\d+)_img(\d+)", img_path.name)
                            if m:
                                page_num = int(m.group(1))
                        except Exception:
                            page_num = None
                        image_info = {
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
                        extracted_images.append(image_info)
                    except Exception as e:
                        logger.warning(f"PDF 이미지 정보 수집 실패: {str(e)}")

                if extracted_images:
                    logger.info(f"   📁 temp_output_dir에서 {len(extracted_images)}개 이미지 발견")

        # 이미지를 영구 위치로 복사
        if extracted_images:
            permanent_images_dir = Path("static/images/pdf")
            permanent_images_dir.mkdir(parents=True, exist_ok=True)

            for image_info in extracted_images:
                src_path = Path(image_info["path"])
                dst_path = permanent_images_dir / image_info["filename"]

                try:
                    import shutil

                    shutil.copy2(src_path, dst_path)
                    # 경로 업데이트
                    image_info["path"] = str(dst_path)
                    image_info["relative_path"] = f"static/images/pdf/{image_info['filename']}"
                    logger.info(f"   🖼️  PDF 이미지 이동: {image_info['filename']}")
                except Exception as e:
                    logger.warning(f"PDF 이미지 이동 실패: {str(e)}")

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

        # MD 파일 저장 (전처리 확인용)
        try:
            md_dir = Path("converted_docs")
            md_dir.mkdir(parents=True, exist_ok=True)

            pdf_name = Path(file_path).stem
            md_file_path = md_dir / f"{pdf_name}.md"

            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(f"# {pdf_name}\n\n")
                f.write(f"**원본 파일**: {Path(file_path).name}\n")
                f.write(f"**변환 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**처리 방법**: {method_name}\n")
                f.write(f"**추출된 이미지**: {len(extracted_images)}개\n\n")
                f.write("---\n\n")
                f.write(markdown_content)

            # 메타데이터에 MD 파일 경로 추가
            document.metadata["md_file_path"] = str(md_file_path)
            document.metadata["md_saved"] = True

            logger.info(f"   📝 MD 파일 저장: {md_file_path}")

            # 2단계 후처리 실행 (설정이 활성화된 경우)
            if settings.enable_md_postprocessing:
                try:
                    logger.info("🔧 2단계 MD 후처리 시작...")

                    # MDPostProcessor 초기화
                    # 에이전트/후처리도 현재 선택된 제공자/모델을 따르도록 동기화
                    _prov = None
                    _model = None
                    try:
                        import streamlit as st  # type: ignore

                        _prov = st.session_state.get("current_provider", None)
                        _model = st.session_state.get("current_model", None)
                    except Exception as err:
                        logger.debug(f"세션 provider/model 조회 실패(무시): {err}")
                    # 후처리 전용 오버라이드 우선 적용(.env): MD_POSTPROCESS_PROVIDER/MODEL
                    _md_override_provider = getattr(settings, "md_postprocess_provider", None)
                    _md_override_model = getattr(settings, "md_postprocess_model", None)
                    if _md_override_provider:
                        _prov = _md_override_provider
                        if _md_override_model:
                            _model = _md_override_model
                    # 그 다음 세션/전역 기본값
                    _prov = _prov or getattr(settings, "llm_provider", "openrouter")
                    if not _model:
                        if _prov == "openai":
                            _model = getattr(settings, "openai_model", None)
                        elif _prov == "google":
                            _model = getattr(settings, "google_model", None)
                        elif _prov == "anthropic":
                            _model = getattr(settings, "anthropic_model", None)
                        elif _prov == "openrouter":
                            # 텍스트용 openrouter_model 우선, 없으면 멀티모달 기본 사용
                            _model = getattr(settings, "openrouter_model", None) or getattr(
                                settings, "openrouter_mm_model", None
                            )
                        else:
                            _model = None

                    postprocessor = MDPostProcessor(
                        output_dir="processed_docs",
                        target_quality=settings.md_postprocess_target_quality,
                        provider=_prov,
                        model_name=_model,
                    )

                    # 품질 검사기 초기화 및 연결(동일 제공자/모델)
                    quality_checker = QualityChecker(provider=_prov, model_name=_model)
                    postprocessor.set_quality_checker(quality_checker)

                    # 2단계 후처리 실행
                    processed_path = postprocessor.process_file(str(md_file_path))

                    if processed_path:
                        logger.info(f"   ✅ 2단계 후처리 완료: {processed_path}")

                        # 처리된 내용으로 document 업데이트
                        with open(processed_path, "r", encoding="utf-8") as f:
                            processed_content = f.read()
                            document.page_content = processed_content

                        document.metadata["postprocessed"] = True
                        document.metadata["processed_md_path"] = str(processed_path)
                        document.metadata["processing_quality"] = postprocessor.last_quality_score
                    else:
                        logger.warning("2단계 후처리 실패 - 원본 유지")
                        document.metadata["postprocessed"] = False

                except Exception as e:
                    logger.error(f"2단계 후처리 중 오류: {str(e)}")
                    document.metadata["postprocessed"] = False
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

        # 지능형 이미지 추출이 활성화된 경우
        if self.use_intelligent_image_extraction:
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

                extraction_results = None
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
                    # 저장된 이미지 정보 수집
                    saved_images = []
                    extracted_texts = []

                    for img in extraction_results["images"]:
                        if img.get("saved"):
                            # 이미지 정보 저장
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

                        # OCR로 추출된 텍스트 수집
                        if img.get("extracted_text"):
                            extracted_texts.append(img["extracted_text"])

                    # 보고서 생성을 위한 마크다운 콘텐츠
                    markdown_content = f"# {Path(file_path).name}\n\n"
                    markdown_content += f"**문서 주제**: {extraction_results.get('document_topic', {}).get('main_topic', '알 수 없음')}\n\n"

                    if saved_images:
                        markdown_content += "## 추출된 이미지\n\n"
                        for img_info in saved_images:
                            markdown_content += f"- 페이지 {img_info['page']}: {img_info['filename']} (관련도: {img_info['relevance_score']:.2f})\n"
                        markdown_content += "\n"

                    if extracted_texts:
                        markdown_content += "## OCR 추출 텍스트\n\n"
                        markdown_content += "\n\n".join(extracted_texts)

                    # 메타데이터 생성
                    {
                        "source": file_path,
                        "processing_method": "intelligent_extraction",
                        "document_topic": extraction_results.get("document_topic", {}),
                        "total_images": extraction_results.get("statistics", {}).get("total_images_found", 0),
                        "relevant_images": extraction_results.get("statistics", {}).get("relevant_images_saved", 0),
                        "text_images_converted": extraction_results.get("statistics", {}).get(
                            "text_images_converted", 0
                        ),
                        "extracted_images": saved_images,
                        "image_extraction_dir": str(output_dir),
                    }

                    # 이미지 추출 정보를 메타데이터에 저장 (나중에 병합용)
                    image_extraction_metadata = {
                        "intelligent_extraction_completed": True,
                        "document_topic": extraction_results.get("document_topic", {}),
                        "total_images": extraction_results.get("statistics", {}).get("total_images_found", 0),
                        "relevant_images": extraction_results.get("statistics", {}).get("relevant_images_saved", 0),
                        "text_images_converted": extraction_results.get("statistics", {}).get(
                            "text_images_converted", 0
                        ),
                        "extracted_images": saved_images,
                        "image_extraction_dir": str(output_dir),
                        "image_markdown_content": markdown_content,
                    }

                    # 추출 보고서 로그
                    logger.info(f"✅ 지능형 이미지 추출 완료: {file_path}")
                    logger.info(f"   📁 이미지 저장 위치: {output_dir}")
                    logger.info(f"   🖼️  관련 이미지: {len(saved_images)}개 저장됨")

                    if progress_callback:
                        progress_callback(
                            0.3, f"지능형 이미지 추출 완료! (관련 이미지 {len(saved_images)}개), PDF 텍스트 처리 중..."
                        )

                    # 이미지 추출 정보를 저장하고 텍스트 처리 계속
                    self.image_extraction_metadata = image_extraction_metadata

                logger.info("지능형 이미지 추출 완료, 기존 방식으로 텍스트 추출 진행")

            except ImportError as e:
                logger.warning(f"이미지 추출 서비스를 사용할 수 없습니다: {str(e)}")
            except Exception as e:
                logger.warning(f"지능형 이미지 추출 실패, 기존 방식으로 폴백: {str(e)}")

        # 에이전트 모드가 활성화된 경우
        if self.use_agent_preprocessing:
            try:
                if progress_callback:
                    progress_callback(0.1, "🤖 에이전트 기반 고품질 변환 중...")

                # 임시 출력 디렉토리 사용
                import tempfile

                temp_output_dir = tempfile.mkdtemp(prefix="agent_pdf_convert_")
                # 에이전트 LLM 제공자/모델을 UI 선택값 또는 설정으로 강제 동기화
                from config import settings as _settings

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
                # 우선순위: 전처리 세션/인자 → 설정 기본
                provider = (provider or getattr(_settings, "llm_provider", "openrouter")).lower()
                if not model:
                    # 설정에서 제공자별 기본 모델 추론
                    if provider == "openai":
                        model = getattr(_settings, "openai_model", None)
                    elif provider == "google":
                        model = getattr(_settings, "google_model", None)
                    elif provider == "anthropic":
                        model = getattr(_settings, "anthropic_model", None)
                    elif provider == "openrouter":
                        model = getattr(_settings, "openrouter_model", None) or getattr(
                            _settings, "openrouter_mm_model", None
                        )
                    else:
                        model = None

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

                        # 에이전트 형식 {page_num: [(path, description), ...]} 을
                        # 지능형 추출 형식 [{'filename': ..., 'path': ..., 'page': ..., 'description': ...}, ...] 으로 변환
                        extracted_images = []
                        for page_num, images in agent_converter.extracted_images_info.items():
                            for img_path, img_description in images:
                                # 상대 경로를 절대 경로로 변환
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

                        # 지능형 이미지 추출 메타데이터 형식으로 저장
                        self.image_extraction_metadata = {
                            "extracted_images": extracted_images,
                            "total_images": len(extracted_images),
                            "relevant_images": len(extracted_images),
                            "text_images_converted": 0,
                            "image_extraction_dir": os.path.join(temp_output_dir, "images"),
                            "document_topic": {"main_topic": "PDF 문서", "keywords": []},
                        }
                        logger.info(f"   ✅ 에이전트 이미지 메타데이터 변환 완료: {len(extracted_images)}개")

                    # 기존 이미지 처리 로직 재사용
                    return self._process_converted_content(
                        markdown_content,
                        file_path,
                        temp_output_dir,
                        progress_callback,
                        processing_method="agent_based_converter",
                    )

            except Exception as e:
                logger.warning(f"에이전트 변환 실패, 기존 방식으로 폴백: {str(e)}")
                # 에이전트 실패 시 기존 방식으로 폴백

        # 기존 방식: 개선된 PDF 변환기 (문장 연결성 향상)
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
            else:
                logger.warning(f"개선된 PDF 변환기에서 내용 추출 실패: {file_path}")
                # 임시 디렉토리 정리
                import shutil

                try:
                    shutil.rmtree(temp_output_dir)
                except Exception as err:
                    logger.debug(f"임시 출력 디렉터리 정리 실패(무시): {err}")

        except Exception as e:
            logger.warning(f"개선된 PDF 변환기 실패, OCR 모드로 전환: {str(e)}")
            # 임시 디렉토리 정리
            import shutil

            try:
                shutil.rmtree(temp_output_dir)
            except Exception as err:
                logger.debug(f"임시 출력 디렉터리 정리 실패(무시): {err}")
            if progress_callback:
                progress_callback(0.3, "OCR 모드로 전환 중...")

        # 2차 시도: OCR을 사용한 고급 PDF 로더
        if self.use_ocr:
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

        # 3차 시도: 기본 PDF 로더 (최후 수단)
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

    def _load_docx_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """DOCX 파일 로딩 및 이미지 추출"""
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx 라이브러리가 설치되지 않았습니다. 'pip install python-docx' 명령으로 설치해주세요."
            )

        try:
            file_name = Path(file_path).name
            file_stem = Path(file_path).stem

            if progress_callback:
                progress_callback(0.1, "DOCX 파일 열기 중...")

            # DOCX 파일 로드
            doc = DocxDocument(file_path)

            if progress_callback:
                progress_callback(0.3, "텍스트 및 이미지 추출 중...")

            # 이미지 저장 디렉토리 생성
            images_dir = Path("static/images/docx")
            images_dir.mkdir(parents=True, exist_ok=True)

            # 이미지 추출
            image_count = 0
            extracted_images = []

            try:
                # 이미지 추출 처리
                image_count, extracted_images = self._extract_docx_images(doc, file_stem, images_dir)
                if progress_callback:
                    progress_callback(0.4, f"이미지 추출 완료 ({image_count}개)")
            except Exception as e:
                logger.warning(f"DOCX 이미지 추출 중 오류: {str(e)}")
                image_count = 0
                extracted_images = []

            if progress_callback:
                progress_callback(0.5, "텍스트 추출 중...")

            # 텍스트 추출
            full_text = []
            paragraph_count = 0

            # 단락별로 텍스트 추출
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:  # 빈 단락 제외
                    full_text.append(text)
                    paragraph_count += 1

            if progress_callback:
                progress_callback(0.7, f"텍스트 정리 중... ({paragraph_count}개 단락)")

            # 표(table) 내용도 추출
            table_count = 0
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        table_text.append(" | ".join(row_text))

                if table_text:
                    full_text.append("\n".join(table_text))
                    table_count += 1

            # 이미지 참조를 텍스트에 추가
            if extracted_images:
                full_text.append("\n## 추출된 이미지")
                for i, image_info in enumerate(extracted_images, 1):
                    full_text.append(f"![이미지 {i}]({image_info['relative_path']})")

            if progress_callback:
                progress_callback(
                    0.9, f"처리 완료 ({paragraph_count}개 단락, {table_count}개 표, {image_count}개 이미지)"
                )

            # 전체 내용 결합
            content = "\n\n".join(full_text)

            if not content.strip():
                raise ValueError("DOCX 파일에서 텍스트를 추출할 수 없습니다.")

            logger.info(f"   📖 DOCX 로딩 완료: {paragraph_count}개 단락, {table_count}개 표, {image_count}개 이미지")

            # Document 객체 생성
            document = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_name": file_name,
                    "file_type": ".docx",
                    "paragraph_count": paragraph_count,
                    "table_count": table_count,
                    "image_count": image_count,
                    "images": extracted_images,
                    "original_size": len(content),
                    "processing_method": "docx_with_images",
                },
            )

            # MD 파일 저장 (전처리 확인용)
            try:
                md_dir = Path("converted_docs")
                md_dir.mkdir(parents=True, exist_ok=True)

                docx_name = Path(file_path).stem
                md_file_path = md_dir / f"{docx_name}.md"

                with open(md_file_path, "w", encoding="utf-8") as f:
                    f.write(f"# {docx_name}\n\n")
                    f.write(f"**원본 파일**: {file_name}\n")
                    f.write(f"**변환 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"**단락 수**: {paragraph_count}개\n")
                    f.write(f"**표 수**: {table_count}개\n")
                    f.write(f"**추출된 이미지**: {image_count}개\n\n")
                    f.write("---\n\n")
                    f.write(content)

                # 메타데이터에 MD 파일 경로 추가
                document.metadata["md_file_path"] = str(md_file_path)
                document.metadata["md_saved"] = True

                logger.info(f"   📝 MD 파일 저장: {md_file_path}")

            except Exception as e:
                logger.warning(f"MD 파일 저장 실패: {str(e)}")
                document.metadata["md_saved"] = False

            if progress_callback:
                progress_callback(1.0, "DOCX 처리 완료")

            return [document]

        except Exception as e:
            logger.error(f"DOCX 파일 로딩 실패: {str(e)}")
            # 폴백으로 텍스트 파일로 처리 시도 (일부 내용이라도 추출)
            try:
                logger.info("기본 텍스트 추출 방법으로 재시도...")
                return self._load_text_file(file_path, progress_callback)
            except Exception:
                raise ValueError(f"DOCX 파일 처리 실패: {str(e)}")

    def _extract_docx_images(self, doc, file_stem: str, images_dir: Path) -> Tuple[int, List[Dict]]:
        """
        DOCX 문서에서 이미지를 추출하고 저장

        Args:
            doc: python-docx Document 객체
            file_stem: 파일명 (확장자 제외)
            images_dir: 이미지 저장 디렉토리

        Returns:
            Tuple[이미지 개수, 이미지 정보 리스트]
        """
        from datetime import datetime

        extracted_images = []
        image_count = 0

        try:
            # DOCX 파일의 모든 관련 parts 확인
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        # 이미지 데이터 추출
                        image_part = rel.target_part
                        image_data = image_part.blob

                        # 이미지 확장자 추정
                        content_type = getattr(image_part, "content_type", "")
                        if "jpeg" in content_type or "jpg" in content_type:
                            ext = ".jpg"
                        elif "png" in content_type:
                            ext = ".png"
                        elif "gif" in content_type:
                            ext = ".gif"
                        elif "bmp" in content_type:
                            ext = ".bmp"
                        else:
                            # 이미지 시그니처로 확장자 추정
                            if image_data.startswith(b"\xff\xd8"):
                                ext = ".jpg"
                            elif image_data.startswith(b"\x89PNG"):
                                ext = ".png"
                            elif image_data.startswith(b"GIF8"):
                                ext = ".gif"
                            elif image_data.startswith(b"BM"):
                                ext = ".bmp"
                            else:
                                ext = ".png"  # 기본값

                        # 파일명 생성
                        image_filename = f"{file_stem}_img{image_count + 1}{ext}"
                        image_path = images_dir / image_filename

                        # 이미지 저장
                        with open(image_path, "wb") as f:
                            f.write(image_data)

                        # 이미지 정보 저장
                        image_info = {
                            "filename": image_filename,
                            "path": str(image_path),
                            "relative_path": f"static/images/docx/{image_filename}",
                            "size": len(image_data),
                            "format": ext[1:].upper(),
                            "content_type": content_type,
                            "extracted_at": datetime.now().isoformat(),
                        }

                        extracted_images.append(image_info)
                        image_count += 1

                        logger.info(f"   🖼️  DOCX 이미지 추출: {image_filename} ({len(image_data)} bytes)")

                    except Exception as e:
                        logger.warning(f"개별 이미지 추출 실패: {str(e)}")
                        continue

            # 인라인 이미지도 확인 (inline shapes)
            try:
                for shape in doc.inline_shapes:
                    if hasattr(shape, "_inline") and hasattr(shape._inline, "graphic"):
                        # 인라인 이미지 처리 (추가 구현 가능)
                        pass
            except Exception as e:
                logger.debug(f"인라인 이미지 처리 중 오류: {str(e)}")

        except Exception as e:
            logger.error(f"DOCX 이미지 추출 실패: {str(e)}")
            return 0, []

        return image_count, extracted_images

    def _preprocess_documents(self, documents: List[Document], file_extension: str = None) -> List[Document]:
        """
        문서 전처리 및 정제
        - 마크다운 파일은 특화 처리
        - 불필요한 내용 제거
        - 텍스트 정규화
        - 한국어 최적화
        - 선택된 전처리 모델 적용
        """
        processed_docs = []

        for doc in documents:
            content = doc.page_content

            # 1. 파일 타입별 특화 전처리
            if file_extension == ".md":
                content = self._clean_markdown_text(content)
            else:
                content = self._clean_text(content)

            # 2. 한국어 특화 정제
            content = self._korean_text_normalization(content)

            # 3. 구조화된 내용 보존
            content = self._preserve_structure(content)

            # 4. 선택된 전처리 모델 적용 (PDF 파일에만 적용)
            if file_extension == ".pdf" and self._preprocessing_model:
                try:
                    logger.info(
                        f"전처리 모델 적용: {self.preprocessing_model} (멀티모달: {self.enable_multimodal_preprocessing})"
                    )

                    if self.enable_multimodal_preprocessing and hasattr(
                        self._preprocessing_model, "preprocess_document_with_images"
                    ):
                        # 멀티모달 전처리
                        logger.info("멀티모달 전처리 수행")
                        # PDF에서 추출된 이미지 정보 가져오기
                        images = []
                        for doc in documents:
                            if hasattr(doc, "metadata") and "images" in doc.metadata:
                                images.extend(doc.metadata["images"])

                        if images:
                            processed_result = self._preprocessing_model.preprocess_document_with_images(
                                content, images
                            )
                            if processed_result and len(processed_result.strip()) > 0:
                                content = processed_result
                                logger.info(f"멀티모달 전처리 완료: {len(content)}자")
                            else:
                                logger.warning("멀티모달 전처리가 빈 결과를 반환했습니다. 일반 전처리로 폴백합니다.")
                                processed_result = self._preprocessing_model.preprocess_text(content)
                                if processed_result and len(processed_result.strip()) > 0:
                                    content = processed_result
                                    logger.info(f"일반 전처리 폴백 완료: {len(content)}자")
                        else:
                            logger.info("이미지가 없어 일반 전처리로 수행")
                            processed_result = self._preprocessing_model.preprocess_text(content)
                            if processed_result and len(processed_result.strip()) > 0:
                                content = processed_result
                                logger.info(f"전처리 모델 적용 완료: {len(content)}자")
                    else:
                        # 일반 전처리
                        processed_result = self._preprocessing_model.preprocess_text(content)
                        if processed_result and len(processed_result.strip()) > 0:
                            content = processed_result
                            logger.info(f"전처리 모델 적용 완료: {len(content)}자")
                        else:
                            logger.warning("전처리 모델이 빈 결과를 반환했습니다. 원본 텍스트를 사용합니다.")

                except Exception as e:
                    logger.error(f"전처리 모델 적용 실패: {e}. 원본 텍스트를 사용합니다.")

            # 5. 너무 짧은 내용 필터링 (100자 이상)
            if len(content.strip()) >= 100:  # 최소 길이 100자
                processed_docs.append(Document(page_content=content, metadata=doc.metadata))
            else:
                logger.debug(f"너무 짧은 콘텐츠 필터링: {len(content)}자 - {content[:50]}...")

        return processed_docs

    def _clean_markdown_text(self, text: str) -> str:
        """마크다운 특화 텍스트 정제 (rebuild_markdown_vector_db.py와 동일한 로직)"""
        if not text:
            return ""

        # 기본 정제
        text = text.strip()

        # 마크다운 메타데이터 제거 (YAML front matter)
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

        # 불필요한 마크다운 구문 정리 (내용은 보존하되 구문만 정리)
        # 이미지 링크는 텍스트만 추출
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # 링크는 텍스트만 추출
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # 코드 블록 표시 제거 (내용은 유지)
        text = re.sub(r"```[a-zA-Z]*\n", "", text)
        text = text.replace("```", "")

        # 인라인 코드 표시 제거
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # 강조 표시 제거 (내용은 유지)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # 굵은 글씨
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # 이탤릭
        text = re.sub(r"__([^_]+)__", r"\1", text)  # 굵은 글씨
        text = re.sub(r"_([^_]+)_", r"\1", text)  # 이탤릭

        # 제목 표시 정리 (# 기호 제거하되 제목은 유지)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # 목록 표시 정리
        text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

        # 일반적인 텍스트 정제
        text = self._general_text_cleaning(text)

        return text

    def _general_text_cleaning(self, text: str) -> str:
        """일반적인 텍스트 정제"""
        # 특수 문자 정리
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\xa0", " ")  # Non-breaking space
        text = text.replace("\u3000", " ")  # Ideographic space

        # 연속된 공백 제거
        text = re.sub(r" +", " ", text)

        # 연속된 줄바꿈 정리 (3개 이상을 2개로)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # 제어 문자 제거
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 한국어 문장 부호 정규화
        text = text.replace("．", ".")
        text = text.replace("，", ",")
        text = text.replace("；", ";")
        text = text.replace("：", ":")

        return text.strip()

    def _clean_text(self, text: str) -> str:
        """기본 텍스트 정제"""
        if not text:
            return ""

        # 연속된 공백 및 줄바꿈 정리
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # 3개 이상의 연속 줄바꿈을 2개로
        text = re.sub(r" +", " ", text)  # 연속된 공백을 하나로

        # 특수 문자 정리
        text = text.replace("\u200b", "")  # Zero-width space
        text = text.replace("\ufeff", "")  # BOM
        text = text.replace("\xa0", " ")  # Non-breaking space
        text = text.replace("\u3000", " ")  # Ideographic space

        # 이상한 문자 제거 (제어 문자 등)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        return text.strip()

    def _korean_text_normalization(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        # 한글 자모 결합 문제 해결
        text = re.sub(r"([ㄱ-ㅎ])([ㅏ-ㅣ])", r"\1\2", text)

        # 한국어 문장 부호 정규화
        text = text.replace("．", ".")
        text = text.replace("，", ",")
        text = text.replace("；", ";")
        text = text.replace("：", ":")

        # 반복되는 특수문자 제거
        text = re.sub(r"[─]{2,}", "─", text)
        text = re.sub(r"[=]{3,}", "===", text)
        text = re.sub(r"[-]{3,}", "---", text)

        return text

    def _preserve_structure(self, text: str) -> str:
        """문서 구조 보존 (제목, 목록 등)"""
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append("")
                continue

            # 제목 형태 보존 (번호가 있는 제목)
            if re.match(r"^\d+\.?\s+[가-힣]", line):
                processed_lines.append(f"\n{line}\n")
            # 목록 항목 보존
            elif re.match(r"^[•▪▫-]\s+", line):
                processed_lines.append(line)
            # 일반 텍스트
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def _split_documents_optimized(self, documents: List[Document], file_extension: str = None) -> List[Document]:
        """
        최적화된 문서 분할
        - 마크다운 파일은 마크다운 전용 분할기 사용
        - 설정에 따라 의미 기반 또는 일반 분할 선택
        - 한국어 문장 구조 고려
        - 이미지 메타데이터 보존
        """
        # 청킹 전에 원본 문서들의 이미지 메타데이터 수집
        original_image_metadata = {}
        for doc in documents:
            if "images" in doc.metadata and doc.metadata["images"]:
                # 파일명을 키로 사용하여 이미지 메타데이터 저장
                file_name = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
                original_image_metadata[file_name] = {
                    "images": doc.metadata["images"],
                    "image_count": doc.metadata.get("image_count", len(doc.metadata["images"])),
                    "intelligent_extraction_completed": doc.metadata.get("intelligent_extraction_completed", False),
                    "extracted_images": doc.metadata.get("extracted_images", []),
                    "image_extraction_dir": doc.metadata.get("image_extraction_dir", ""),
                }
                logger.info(
                    f"   🖼️  청킹 전 이미지 메타데이터 보존: {file_name} - {len(doc.metadata['images'])}개 이미지"
                )

        # 마크다운 파일은 마크다운 전용 분할기 사용
        if file_extension == ".md":
            logger.info("마크다운 특화 청킹 사용")
            chunks = self.markdown_splitter.split_documents(documents)
            chunks = self._post_process_markdown_chunks(chunks)
        # 기타 파일 타입
        elif settings.use_semantic_chunking and self.semantic_splitter:
            logger.info("의미 기반 청킹 사용")
            try:
                chunks = self.semantic_splitter.split_documents(documents)
                # 의미 기반 청킹 성공 시 후처리
                chunks = self._post_process_semantic_chunks(chunks)
            except Exception as e:
                logger.warning(f"의미 기반 청킹 실패, 기본 청킹 사용: {str(e)}")
                chunks = self._enhanced_default_chunking(documents)
        else:
            # 기본 청킹 (향상된 버전)
            logger.info("향상된 기본 청킹 사용")
            chunks = self._enhanced_default_chunking(documents)

        # 청킹 후 각 청크에 이미지 메타데이터 복원
        if original_image_metadata:
            restored_count = 0
            for chunk in chunks:
                file_name = chunk.metadata.get("file_name", chunk.metadata.get("source", "unknown"))
                if file_name in original_image_metadata:
                    # 원본 이미지 메타데이터를 청크에 추가
                    img_meta = original_image_metadata[file_name]
                    chunk.metadata.update(img_meta)
                    restored_count += 1

            if restored_count > 0:
                logger.info(f"   ✅ 청킹 후 이미지 메타데이터 복원 완료: {restored_count}개 청크")
            else:
                logger.warning("   ⚠️  경고: 이미지 메타데이터 복원 실패 - 파일명 매칭 안됨")

        return chunks

    def _post_process_markdown_chunks(self, chunks: List[Document]) -> List[Document]:
        """마크다운 청킹 후처리 - 짧은 청크 병합 및 품질 개선"""
        processed_chunks = []
        merged_count = 0

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()

            # 최소 길이 체크
            if len(content) < 200:  # 200자 미만은 이전 청크와 병합 시도
                if processed_chunks:
                    last_chunk = processed_chunks[-1]
                    combined_content = last_chunk.page_content + "\n\n" + content

                    # 병합 후 크기가 적절하면 병합
                    if len(combined_content) <= settings.chunk_size * 1.3:
                        processed_chunks[-1] = Document(
                            page_content=combined_content, metadata={**last_chunk.metadata, "merged_chunks": True}
                        )
                        merged_count += 1
                        continue

            # 유효한 청크로 판단
            if len(content) >= 50:  # 최소 50자 이상
                processed_chunks.append(chunk)

        if merged_count > 0:
            logger.info(f"   🔗 마크다운 청크 병합: {merged_count}개")

        return processed_chunks

    def _enhanced_default_chunking(self, documents: List[Document]) -> List[Document]:
        """향상된 기본 청킹"""
        logger.debug(f"청킹 입력: {len(documents)}개 문서")
        for i, doc in enumerate(documents):
            logger.debug(f"  문서 {i}: {len(doc.page_content)}자 - '{doc.page_content[:50]}...'")

        # 한국어에 최적화된 분리자 순서
        korean_optimized_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # 여러 줄바꿈
                "\n\n",  # 문단 분리
                "\n",  # 줄바꿈
                "。",  # 한국어 마침표 (문서에 따라)
                ".",  # 영어 마침표
                "!",  # 느낌표
                "?",  # 물음표
                ";",  # 세미콜론
                ":",  # 콜론
                ",",  # 쉼표
                " ",  # 공백
                "",  # 문자 단위
            ],
        )

        chunks = korean_optimized_splitter.split_documents(documents)
        logger.debug(f"스플리터 결과: {len(chunks)}개 청크")
        for i, chunk in enumerate(chunks):
            logger.debug(f"  청크 {i}: {len(chunk.page_content)}자")

        # 기본 청킹에도 후처리 적용
        processed = self._post_process_semantic_chunks(chunks)
        logger.debug(f"후처리 결과: {len(processed)}개 청크")
        return processed

    def _post_process_semantic_chunks(self, chunks: List[Document]) -> List[Document]:
        """의미 기반 청킹 후처리 - 짧은 청크 병합 강화"""
        logger.debug(f"후처리 시작: {len(chunks)}개 청크 입력")
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            logger.debug(f"  처리 중 청크 {i}: {len(content)}자")

            # 너무 짧은 청크는 이전 청크와 합치기 (500자 미만)
            if len(content) < 500 and processed_chunks:
                logger.debug(f"    → 병합 시도: {len(content)}자 < 500, processed_chunks 있음")
                last_chunk = processed_chunks[-1]
                combined_content = last_chunk.page_content + "\n\n" + content

                # 합쳐도 최대 크기를 넘지 않으면 합치기
                if len(combined_content) <= settings.chunk_size * 1.5:  # 1.2에서 1.5로 증가
                    processed_chunks[-1] = Document(page_content=combined_content, metadata=last_chunk.metadata)
                    logger.debug(f"    ✓ 짧은 청크 병합: {len(content)}자 -> {len(combined_content)}자")
                    continue
                else:
                    # 합쳐도 너무 크면 별도 청크로 추가
                    logger.debug(f"    → 병합 시 크기 초과, 별도 청크로 유지: {len(content)}자")
            elif len(content) < 500:
                logger.debug(f"    → 병합 불가: {len(content)}자 < 500, 하지만 processed_chunks 비어있음")

            # 여전히 너무 짧은 청크는 제외
            if len(content) >= 100:  # 최소 크기 보장 (300 → 100으로 완화)
                processed_chunks.append(chunk)
                logger.debug(f"    ✓ 청크 추가: {len(content)}자 >= 100")
            else:
                logger.debug(f"    ✗ 너무 짧은 청크 제외: {len(content)}자 < 100 - {content[:50]}...")

        logger.debug(f"후처리 완료: {len(processed_chunks)}개 청크 출력")
        return processed_chunks

    def _enhance_metadata(self, chunks: List[Document], file_path: str) -> List[Document]:
        """메타데이터 보강"""
        enhanced_chunks = []
        file_name = Path(file_path).name
        file_extension = Path(file_path).suffix.lower()

        # 디버깅: 첫 번째 청크의 원본 메타데이터 확인
        if chunks and len(chunks) > 0:
            first_chunk_metadata = chunks[0].metadata
            has_images = "images" in first_chunk_metadata
            image_count = len(first_chunk_metadata.get("images", [])) if has_images else 0
            logger.info(f"   🔍 청킹 후 메타데이터 검증: images 필드 존재={has_images}, 이미지 개수={image_count}")
            if not has_images or image_count == 0:
                logger.warning(
                    "   ⚠️  경고: 청킹 후 이미지 메타데이터 손실! 원본 Document의 images 필드가 청크에 전달되지 않았습니다."
                )

        for i, chunk in enumerate(chunks):
            # 기존 메타데이터 복사
            metadata = chunk.metadata.copy()

            # 추가 메타데이터
            metadata.update(
                {
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": file_extension,
                    "chunk_id": f"{file_name}_{i:04d}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "processing_method": metadata.get(
                        "processing_method",
                        "markdown_optimized"
                        if file_extension == ".md"
                        else "semantic"
                        if settings.use_semantic_chunking and self.semantic_splitter
                        else "enhanced_default",
                    ),
                }
            )

            # 디버깅: 개별 청크의 이미지 메타데이터 확인
            if i == 0:  # 첫 번째 청크만 로그
                logger.debug(f"   📝 첫 번째 청크 메타데이터 키: {list(metadata.keys())}")
                if "images" in metadata:
                    logger.debug(f"   🖼️  첫 번째 청크 이미지 개수: {len(metadata['images'])}")

            enhanced_chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

        return enhanced_chunks

    def load_directory(self, directory_path: str) -> List[Document]:
        """디렉토리 내의 모든 문서를 로드하고 청크로 분할"""
        all_documents = []
        supported_extensions = [".txt", ".md", ".pdf", ".docx"]

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        documents = self.load_document(file_path)
                        all_documents.extend(documents)
                        print(f"로드 완료: {file_path} ({len(documents)} 청크)")
                    except Exception as e:
                        print(f"파일 로드 실패: {file_path} - {str(e)}")

        return all_documents

    def process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """문서를 벡터 DB에 저장하기 적합한 형태로 처리"""
        processed_docs = []

        for i, doc in enumerate(documents):
            processed_doc = {
                "id": doc.metadata.get("chunk_id", f"doc_{i}"),
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            processed_docs.append(processed_doc)

        return processed_docs


# 기존 DocumentLoader와의 호환성 유지
class DocumentLoader(EnhancedDocumentLoader):
    """기존 DocumentLoader와의 호환성을 위한 클래스"""

    pass
