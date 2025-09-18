"""
파일 로더 관리 모듈

이 모듈은 다양한 파일 형식의 로딩을 담당합니다.
PDF, DOCX, Markdown, Text 파일을 처리하고 파일 형식 감지 및 라우팅을 수행합니다.
"""

from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path
import os
import logging
import tempfile
from datetime import datetime

from config import settings
from ..utils.pdf_converter import ImprovedPDFConverter
from ..utils.agent_pdf_converter import AgentBasedPDFConverter
from .pdf_loader_advanced import AdvancedPDFLoader

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileLoaderManager:
    """
    파일 로딩 관리자 클래스
    
    주요 기능:
    - 다양한 파일 형식 지원 (PDF, DOCX, Markdown, Text)
    - 파일 형식 자동 감지 및 적절한 로더 라우팅
    - 에이전트 기반 전처리 지원
    - 지능형 이미지 추출 지원
    - 인코딩 자동 감지
    """
    
    def __init__(self, use_ocr: bool = True, use_agent_preprocessing: bool = False,
                 use_intelligent_image_extraction: bool = False, preprocessing_model: str = 'local'):
        self.use_ocr = use_ocr
        self.use_agent_preprocessing = use_agent_preprocessing
        self.use_intelligent_image_extraction = use_intelligent_image_extraction
        self.preprocessing_model = preprocessing_model
        
        # 지능형 이미지 추출 메타데이터 저장용
        self.image_extraction_metadata = None
        
        # 지원 파일 형식
        self.supported_extensions = {
            '.pdf': self._load_pdf_file,
            '.txt': self._load_text_file,
            '.md': self._load_markdown_file,
            '.docx': self._load_docx_file,
            '.doc': self._load_docx_file  # DOCX 로더로 시도
        }
        
        logger.info("FileLoaderManager 초기화 완료")
    
    def load_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """
        파일 로딩 메인 인터페이스
        
        Args:
            file_path: 로딩할 파일 경로
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            Document 객체 리스트
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if progress_callback:
                progress_callback(f"파일 형식 감지: {file_extension}", 0.1)
            
            # 파일 존재 확인
            if not file_path.exists():
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
            
            # 파일 크기 확인
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"파일 로딩 시작: {file_path.name} ({file_size_mb:.1f}MB)")
            
            # 적절한 로더 선택
            if file_extension in self.supported_extensions:
                loader_func = self.supported_extensions[file_extension]
                documents = loader_func(str(file_path), progress_callback)
            else:
                # 지원하지 않는 형식은 텍스트로 시도
                logger.warning(f"지원하지 않는 파일 형식: {file_extension}, 텍스트로 시도")
                documents = self._load_text_file(str(file_path), progress_callback)
            
            if progress_callback:
                progress_callback(f"로딩 완료: {len(documents)}개 문서", 0.9)
            
            logger.info(f"파일 로딩 성공: {file_path.name}, {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logger.error(f"파일 로딩 실패: {file_path} - {str(e)}")
            raise
    
    def _load_markdown_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """
        마크다운 파일 전용 로딩 (인코딩 자동 감지)
        """
        try:
            file_name = Path(file_path).name
            
            # 다양한 인코딩으로 시도
            encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"인코딩 {encoding} 시도 실패: {str(e)}")
                    continue
            
            if content is None:
                raise ValueError("마크다운 파일 로딩 실패: 지원되는 인코딩 없음")
            
            logger.info(f"마크다운 로딩 완료: {used_encoding} 인코딩")
            
            if progress_callback:
                progress_callback("마크다운 텍스트 추출 완료", 0.5)
            
            # Document 객체 생성 (마크다운 특화 메타데이터 포함)
            document = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'file_name': file_name,
                    'file_type': '.md',
                    'encoding': used_encoding,
                    'original_size': len(content),
                    'processing_method': 'markdown_optimized'
                }
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"마크다운 파일 로딩 실패: {str(e)}")
            # 기본 텍스트 로더로 폴백
            return self._load_text_file(file_path, progress_callback)
    
    def _load_text_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """텍스트 파일 로딩 (인코딩 자동 감지)"""
        try:
            file_name = Path(file_path).name
            
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    logger.info(f"파일 인코딩 감지: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("지원되는 인코딩을 찾을 수 없습니다")
            
            if progress_callback:
                progress_callback("텍스트 추출 완료", 0.5)
            
            document = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'file_name': file_name,
                    'file_type': Path(file_path).suffix.lower(),
                    'encoding': used_encoding,
                    'original_size': len(content),
                    'processing_method': 'text_direct'
                }
            )
            
            return [document]
            
        except Exception as e:
            logger.error(f"텍스트 파일 로딩 실패: {str(e)}")
            # 기본 로더로 폴백
            try:
                loader = TextLoader(file_path)
                documents = loader.load()
                return documents
            except Exception as fallback_error:
                logger.error(f"기본 로더도 실패: {fallback_error}")
                raise
    
    def _load_pdf_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """PDF 파일 로딩 - 에이전트 모드, 지능형 이미지 추출, 또는 개선된 PDF 변환기 사용"""
        
        # 중복 업로드 대비: 동일 문서의 기존 이미지 정리
        try:
            self._cleanup_existing_images_for_pdf(file_path)
        except Exception as cleanup_err:
            logger.warning(f"기존 이미지 정리 중 경고: {str(cleanup_err)}")

        # 각 파일 처리 시작 시 메타데이터 초기화
        self.image_extraction_metadata = None
        
        # 지능형 이미지 추출이 활성화된 경우
        if self.use_intelligent_image_extraction:
            try:
                if progress_callback:
                    progress_callback("🧠 지능형 이미지 추출 중...", 0.1)
                else:
                    logger.info("🧠 지능형 이미지 추출 시작 (provider=%s)", settings.image_analysis_provider.lower())
                
                # PDF 파일명 기반으로 출력 디렉토리 생성
                pdf_name = Path(file_path).stem
                output_base_dir = Path("data/extracted_images")
                output_dir = output_base_dir / pdf_name
                output_dir.mkdir(parents=True, exist_ok=True)

                extraction_results = None
                # 1순위: 설정된 이미지 분석 프로바이더
                if settings.image_analysis_provider.lower() == "openrouter":
                    try:
                        from ..utils.openrouter_image_service import OpenRouterImageService
                        service = OpenRouterImageService()
                        extraction_results = service.extract_and_analyze_pdf_images(
                            file_path, str(output_dir)
                        )
                        if extraction_results:
                            logger.info(f"✅ OpenRouter 이미지 추출 완료: {extraction_results.get('total_images', 0)}개")
                        else:
                            logger.warning("⚠️ OpenRouter 이미지 추출 결과가 비어있음")
                    except Exception as e:
                        logger.warning(f"❌ OpenRouter 이미지 추출 실패: {str(e)}")
                        extraction_results = None

                # 2순위: 로컬 OCR 기반 (pymupdf + OCR)
                if not extraction_results:
                    try:
                        from ..utils.pymupdf_image_extractor import PyMuPDFImageExtractor
                        extractor = PyMuPDFImageExtractor()
                        extraction_results = extractor.extract_images(file_path, str(output_dir))
                        if extraction_results:
                            logger.info(f"✅ PyMuPDF 이미지 추출 완료: {extraction_results.get('total_images', 0)}개")
                    except Exception as e:
                        logger.warning(f"❌ PyMuPDF 이미지 추출 실패: {str(e)}")
                        extraction_results = None

                # 이미지 추출 메타데이터 저장
                if extraction_results:
                    self.image_extraction_metadata = {
                        'extraction_method': extraction_results.get('extraction_method', 'unknown'),
                        'total_images': extraction_results.get('total_images', 0),
                        'successful_extractions': extraction_results.get('successful_extractions', 0),
                        'image_analysis_results': extraction_results.get('image_analysis_results', []),
                        'output_directory': str(output_dir)
                    }
                    logger.info(f"📊 이미지 추출 메타데이터 저장 완료")
                else:
                    logger.warning("⚠️ 모든 이미지 추출 방법 실패")

                if progress_callback:
                    progress_callback("이미지 추출 완료, PDF 텍스트 처리 중...", 0.3)

            except Exception as image_error:
                logger.error(f"❌ 지능형 이미지 추출 중 오류: {str(image_error)}")
                if progress_callback:
                    progress_callback("이미지 추출 실패, PDF 텍스트 처리 중...", 0.3)

        # 에이전트 기반 전처리가 활성화된 경우
        if self.use_agent_preprocessing:
            try:
                if progress_callback:
                    progress_callback("🤖 에이전트 기반 PDF 변환 중...", 0.4)
                
                converter = AgentBasedPDFConverter(model=self.preprocessing_model)
                
                # 임시 디렉토리 생성
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_output_path = Path(temp_dir) / f"{Path(file_path).stem}.md"
                    
                    # 에이전트 기반 변환 실행
                    success = converter.convert_pdf_to_markdown(
                        str(file_path), 
                        str(temp_output_path),
                        progress_callback=progress_callback
                    )
                    
                    if success and temp_output_path.exists():
                        # 변환된 마크다운 읽기
                        content = temp_output_path.read_text(encoding='utf-8')
                        
                        document = Document(
                            page_content=content,
                            metadata={
                                'source': str(file_path),
                                'file_name': Path(file_path).name,
                                'file_type': '.pdf',
                                'processing_method': 'agent_based_conversion',
                                'preprocessing_model': self.preprocessing_model,
                                'original_size': len(content)
                            }
                        )
                        
                        # 이미지 추출 메타데이터가 있으면 추가
                        if self.image_extraction_metadata:
                            document.metadata.update({
                                'image_extraction_metadata': self.image_extraction_metadata
                            })
                        
                        logger.info("에이전트 기반 PDF 변환 성공")
                        return [document]
                    else:
                        logger.warning("에이전트 기반 변환 실패, 기본 PDF 로더로 폴백")
                        
            except Exception as e:
                logger.error(f"에이전트 기반 PDF 변환 실패: {str(e)}")
                if progress_callback:
                    progress_callback("에이전트 변환 실패, 기본 PDF 처리 중...", 0.4)
        
        # 기본 PDF 로딩 로직
        try:
            if progress_callback:
                progress_callback("📄 PDF 텍스트 추출 중...", 0.5)
            
            # 1순위: AdvancedPDFLoader 시도
            try:
                advanced_loader = AdvancedPDFLoader(use_ocr=self.use_ocr)
                documents = advanced_loader.load_pdf(file_path, progress_callback)
                
                if documents and len(documents) > 0:
                    # 메타데이터 강화
                    for doc in documents:
                        doc.metadata.update({
                            'file_name': Path(file_path).name,
                            'file_type': '.pdf',
                            'processing_method': 'advanced_pdf_loader',
                            'ocr_enabled': self.use_ocr
                        })
                        
                        # 이미지 추출 메타데이터가 있으면 추가
                        if self.image_extraction_metadata:
                            doc.metadata.update({
                                'image_extraction_metadata': self.image_extraction_metadata
                            })
                    
                    logger.info(f"AdvancedPDFLoader 성공: {len(documents)}개 문서")
                    return documents
                    
            except Exception as e:
                logger.warning(f"AdvancedPDFLoader 실패: {str(e)}")
            
            # 2순위: 개선된 PDF 변환기 시도
            try:
                if progress_callback:
                    progress_callback("🔄 개선된 PDF 변환기 시도 중...", 0.6)
                
                converter = ImprovedPDFConverter()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_output_path = Path(temp_dir) / f"{Path(file_path).stem}.md"
                    
                    success = converter.convert_pdf_to_markdown(
                        str(file_path), 
                        str(temp_output_path)
                    )
                    
                    if success and temp_output_path.exists():
                        content = temp_output_path.read_text(encoding='utf-8')
                        
                        document = Document(
                            page_content=content,
                            metadata={
                                'source': str(file_path),
                                'file_name': Path(file_path).name,
                                'file_type': '.pdf',
                                'processing_method': 'improved_pdf_converter',
                                'original_size': len(content)
                            }
                        )
                        
                        # 이미지 추출 메타데이터가 있으면 추가
                        if self.image_extraction_metadata:
                            document.metadata.update({
                                'image_extraction_metadata': self.image_extraction_metadata
                            })
                        
                        logger.info("개선된 PDF 변환기 성공")
                        return [document]
                        
            except Exception as e:
                logger.warning(f"개선된 PDF 변환기 실패: {str(e)}")
            
            # 3순위: 기본 PyPDFLoader 시도
            if progress_callback:
                progress_callback("📖 기본 PDF 로더 시도 중...", 0.7)
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # 메타데이터 강화
            for doc in documents:
                doc.metadata.update({
                    'file_name': Path(file_path).name,
                    'file_type': '.pdf',
                    'processing_method': 'pypdf_loader'
                })
                
                # 이미지 추출 메타데이터가 있으면 추가
                if self.image_extraction_metadata:
                    doc.metadata.update({
                        'image_extraction_metadata': self.image_extraction_metadata
                    })
            
            logger.info(f"기본 PDF 로더 성공: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logger.error(f"모든 PDF 로딩 방법 실패: {str(e)}")
            raise
    
    def _load_docx_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """DOCX 파일 로딩 및 이미지 추출"""
        
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx 패키지가 설치되지 않았습니다. pip install python-docx로 설치하세요.")
        
        try:
            if progress_callback:
                progress_callback("📄 DOCX 파일 처리 중...", 0.2)
            
            # DOCX 문서 열기
            doc = DocxDocument(file_path)
            
            # 텍스트 추출
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text)
            
            content = '\n'.join(paragraphs)
            
            if progress_callback:
                progress_callback("📝 텍스트 추출 완료, 이미지 처리 중...", 0.6)
            
            # 이미지 추출
            image_count, extracted_images = self._extract_docx_images(doc, Path(file_path).stem)
            
            if progress_callback:
                progress_callback(f"🖼️ 이미지 추출 완료: {image_count}개", 0.8)
            
            # 메타데이터 생성
            metadata = {
                'source': str(file_path),
                'file_name': Path(file_path).name,
                'file_type': '.docx',
                'processing_method': 'python_docx',
                'original_size': len(content),
                'paragraph_count': len(paragraphs),
                'image_count': image_count
            }
            
            # 추출된 이미지 정보가 있으면 메타데이터에 추가
            if extracted_images:
                metadata['extracted_images'] = extracted_images
            
            document = Document(page_content=content, metadata=metadata)
            
            logger.info(f"DOCX 로딩 성공: {Path(file_path).name}, 이미지 {image_count}개")
            return [document]
            
        except Exception as e:
            logger.error(f"DOCX 파일 로딩 실패: {str(e)}")
            raise
    
    def _extract_docx_images(self, doc, file_stem: str) -> Tuple[int, List[Dict]]:
        """DOCX 파일에서 이미지 추출"""
        try:
            import zipfile
            from docx.oxml.ns import nsdecls
            from docx.oxml import parse_xml
            
            # 이미지 저장 디렉토리 생성
            images_dir = Path("static/images/docx")
            images_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_images = []
            image_count = 0
            
            # DOCX 파일을 ZIP으로 열어 이미지 추출
            doc_path = doc.core_properties.created  # 문서 경로 대신 생성일로 대체
            
            # 문서 내 이미지 관계 찾기
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        # 이미지 데이터 가져오기
                        image_data = rel.target_part.blob
                        
                        # 파일 확장자 결정
                        content_type = rel.target_part.content_type
                        ext_map = {
                            'image/jpeg': '.jpg',
                            'image/png': '.png',
                            'image/gif': '.gif',
                            'image/bmp': '.bmp'
                        }
                        ext = ext_map.get(content_type, '.jpg')
                        
                        # 파일명 생성
                        image_filename = f"{file_stem}_image_{image_count + 1}{ext}"
                        image_path = images_dir / image_filename
                        
                        # 이미지 파일 저장
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_data)
                        
                        # 이미지 정보 저장
                        image_info = {
                            'filename': image_filename,
                            'path': str(image_path),
                            'relative_path': f"static/images/docx/{image_filename}",
                            'size': len(image_data),
                            'format': ext[1:].upper(),
                            'content_type': content_type,
                            'extracted_at': datetime.now().isoformat()
                        }
                        
                        extracted_images.append(image_info)
                        image_count += 1
                        
                        logger.info(f"DOCX 이미지 추출: {image_filename} ({len(image_data)} bytes)")
                        
                    except Exception as e:
                        logger.warning(f"개별 이미지 추출 실패: {str(e)}")
                        continue
            
        except Exception as e:
            logger.error(f"DOCX 이미지 추출 실패: {str(e)}")
            return 0, []
        
        return image_count, extracted_images
    
    def _cleanup_existing_images_for_pdf(self, file_path: str) -> None:
        """PDF와 연관된 기존 이미지 정리"""
        try:
            pdf_name = Path(file_path).stem
            
            # 가능한 이미지 디렉토리들
            image_dirs = [
                Path("data/extracted_images") / pdf_name,
                Path("static/images/pdf") / pdf_name,
                Path("temp_images") / pdf_name
            ]
            
            for img_dir in image_dirs:
                if img_dir.exists() and img_dir.is_dir():
                    # 디렉토리 내 파일들 삭제
                    for file in img_dir.iterdir():
                        if file.is_file():
                            file.unlink()
                            logger.debug(f"기존 이미지 삭제: {file}")
                    
                    # 빈 디렉토리 삭제 시도
                    try:
                        img_dir.rmdir()
                        logger.debug(f"빈 디렉토리 삭제: {img_dir}")
                    except OSError:
                        # 디렉토리가 비어있지 않은 경우 무시
                        pass
                        
        except Exception as e:
            logger.warning(f"이미지 정리 중 오류: {str(e)}")
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return list(self.supported_extensions.keys())
    
    def is_supported_file(self, file_path: str) -> bool:
        """파일이 지원되는 형식인지 확인"""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 정보 반환"""
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                'name': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'is_supported': self.is_supported_file(file_path),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"파일 정보 조회 실패: {str(e)}")
            return {}