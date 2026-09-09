from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import os
from pathlib import Path
from config import settings
from .pdf_loader_advanced import AdvancedPDFLoader
from .text_cleaning import TextCleaningMixin
from .chunking import ChunkingMixin
from .docx_loading import DocxLoadingMixin
from .pdf_loading import PdfLoadingMixin
import logging

logger = logging.getLogger(__name__)


class EnhancedDocumentLoader(TextCleaningMixin, ChunkingMixin, DocxLoadingMixin, PdfLoadingMixin):
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
                    logger.debug(f"   인코딩 {encoding} 디코딩 실패, 다음 후보 시도")
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
                    logger.debug(f"   인코딩 {encoding} 디코딩 실패, 다음 후보 시도")
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
