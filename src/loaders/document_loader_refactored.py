"""
리팩터링된 문서 로더 메인 오케스트레이터

기존 1,427줄의 monolithic document_loader.py를 모듈화된 구조로 리팩터링한 메인 오케스트레이터입니다.
새로 생성된 모듈들을 조합하여 전체 문서 로딩 파이프라인을 관리합니다.

모듈 구성:
- FileLoaderManager: 파일 포맷별 로딩 및 라우팅
- TextProcessor: 텍스트 정제 및 정규화
- DocumentSplitter: 문서 분할 및 청킹
- ContentProcessor: 전처리 오케스트레이션 및 에이전트 처리
"""

from typing import List, Dict, Any, Optional, Callable
from langchain.schema import Document
from pathlib import Path
import logging
import time

# 새로 생성한 모듈들 임포트
from .file_loader_manager import FileLoaderManager
from .text_processor import TextProcessor
from .document_splitter import DocumentSplitter
from .content_processor import ContentProcessor

# 설정 및 기존 모듈들
try:
    from config import settings
except ImportError:
    # 설정이 없는 경우 기본값 사용
    class DefaultSettings:
        use_ocr = True
        use_agent_preprocessing = False
        enable_postprocessing = False
        use_intelligent_image_extraction = False
        preprocessing_model = 'local'
        enable_multimodal_preprocessing = False
        chunk_size = 1200
        chunk_overlap = 200
        min_chunk_size = 300
    
    settings = DefaultSettings()

logger = logging.getLogger(__name__)

# 성능 유틸
try:
    from src.utils.perf import now
except Exception:
    def now() -> float:
        return time.perf_counter()


class DocumentLoader:
    """
    리팩터링된 문서 로더 메인 오케스트레이터
    
    기존 EnhancedDocumentLoader의 모든 기능을 유지하면서 모듈화된 구조를 사용합니다.
    - FileLoaderManager: 파일 로딩 관리
    - TextProcessor: 텍스트 처리
    - DocumentSplitter: 문서 분할
    - ContentProcessor: 콘텐츠 전처리
    """
    
    def __init__(self, 
                 use_ocr: bool = None,
                 use_agent_preprocessing: bool = None,
                 enable_postprocessing: bool = None,
                 use_intelligent_image_extraction: bool = None,
                 preprocessing_model: str = None,
                 enable_multimodal_preprocessing: bool = None):
        """
        DocumentLoader 초기화
        
        Args:
            use_ocr: OCR 사용 여부
            use_agent_preprocessing: 에이전트 기반 전처리 사용 여부
            enable_postprocessing: 후처리 활성화 여부
            use_intelligent_image_extraction: 지능형 이미지 추출 사용 여부
            preprocessing_model: 전처리 모델 ('local', 'openai', 'google', 'anthropic', 'openrouter')
            enable_multimodal_preprocessing: 멀티모달 전처리 활성화 여부
        """
        # 설정값 초기화 (매개변수가 None이면 settings에서 가져오기)
        self.use_ocr = use_ocr if use_ocr is not None else getattr(settings, 'use_ocr', True)
        self.use_agent_preprocessing = (use_agent_preprocessing if use_agent_preprocessing is not None 
                                      else getattr(settings, 'use_agent_preprocessing', False))
        self.enable_postprocessing = (enable_postprocessing if enable_postprocessing is not None
                                    else getattr(settings, 'enable_postprocessing', False))
        self.use_intelligent_image_extraction = (use_intelligent_image_extraction if use_intelligent_image_extraction is not None
                                               else getattr(settings, 'use_intelligent_image_extraction', False))
        self.preprocessing_model = (preprocessing_model if preprocessing_model is not None
                                  else getattr(settings, 'preprocessing_model', 'local'))
        self.enable_multimodal_preprocessing = (enable_multimodal_preprocessing if enable_multimodal_preprocessing is not None
                                              else getattr(settings, 'enable_multimodal_preprocessing', False))
        
        # 모듈화된 구성 요소들 초기화
        self.file_loader = FileLoaderManager(
            use_ocr=self.use_ocr,
            use_agent_preprocessing=self.use_agent_preprocessing,
            use_intelligent_image_extraction=self.use_intelligent_image_extraction,
            preprocessing_model=self.preprocessing_model
        )
        
        self.text_processor = TextProcessor()
        
        self.document_splitter = DocumentSplitter(
            chunk_size=getattr(settings, 'chunk_size', 1200),
            chunk_overlap=getattr(settings, 'chunk_overlap', 200),
            min_chunk_size=getattr(settings, 'min_chunk_size', 300)
        )
        
        self.content_processor = ContentProcessor() if self.use_agent_preprocessing else None
        
        # 통계 정보
        self.stats = {
            'files_processed': 0,
            'documents_loaded': 0,
            'chunks_created': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
        logger.info(f"DocumentLoader 초기화 완료 - OCR: {self.use_ocr}, "
                   f"에이전트: {self.use_agent_preprocessing}, "
                   f"멀티모달: {self.enable_multimodal_preprocessing}")
    
    def load_documents(self, file_path: str, progress_callback: Optional[Callable] = None) -> List[Document]:
        """
        파일에서 문서 로드 및 처리
        
        Args:
            file_path: 로드할 파일 경로
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            처리된 문서 리스트
        """
        start_time = now()
        
        try:
            logger.info(f"문서 로딩 시작: {file_path}")
            
            # 1단계: 파일 로딩
            if progress_callback:
                progress_callback("파일 로딩 중...", 0.1)
            
            raw_documents = self.file_loader.load_file(file_path, progress_callback)
            
            if not raw_documents:
                logger.warning(f"파일에서 문서를 로드할 수 없습니다: {file_path}")
                return []
            
            logger.info(f"파일 로딩 완료: {len(raw_documents)} 문서")
            
            # 2단계: 텍스트 처리
            if progress_callback:
                progress_callback("텍스트 처리 중...", 0.3)
            
            processed_documents = self.text_processor.process_documents(
                raw_documents,
                clean_text=True,
                normalize_korean=True,
                extract_structure=True,
                optimize_search=True,
                remove_duplicates=True
            )
            
            logger.info(f"텍스트 처리 완료: {len(processed_documents)} 문서")
            
            # 3단계: 콘텐츠 전처리 (에이전트 기반)
            if self.content_processor and progress_callback:
                progress_callback("콘텐츠 전처리 중...", 0.5)
                
                enhanced_documents = self.content_processor.process_documents(
                    processed_documents,
                    file_path=file_path,
                    progress_callback=lambda current, total: progress_callback(
                        f"전처리 중... ({current}/{total})", 
                        0.5 + 0.2 * (current / total)
                    )
                )
                processed_documents = enhanced_documents
                
                logger.info(f"콘텐츠 전처리 완료: {len(processed_documents)} 문서")
            
            # 4단계: 문서 분할
            if progress_callback:
                progress_callback("문서 분할 중...", 0.7)
            
            final_chunks = self.document_splitter.split_documents(
                processed_documents,
                strategy='auto',
                preserve_structure=True
            )
            
            logger.info(f"문서 분할 완료: {len(final_chunks)} 청크")
            
            # 5단계: 후처리 (필요시)
            if self.enable_postprocessing and progress_callback:
                progress_callback("후처리 중...", 0.9)
                final_chunks = self._apply_postprocessing(final_chunks)
                logger.info(f"후처리 완료: {len(final_chunks)} 청크")
            
            # 통계 업데이트
            processing_time = now() - start_time
            self._update_stats(file_path, len(raw_documents), len(final_chunks), processing_time)
            
            if progress_callback:
                progress_callback("완료", 1.0)
            
            logger.info(f"문서 로딩 완료: {file_path} ({processing_time:.2f}초)")
            return final_chunks
            
        except Exception as e:
            error_msg = f"문서 로딩 중 오류: {file_path} - {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            
            if progress_callback:
                progress_callback(f"오류: {str(e)}", 1.0)
            
            return []
    
    def load_multiple_files(self, file_paths: List[str], 
                          progress_callback: Optional[Callable] = None) -> List[Document]:
        """
        여러 파일 일괄 로드
        
        Args:
            file_paths: 로드할 파일 경로 리스트
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            모든 파일에서 로드된 문서 리스트
        """
        all_documents = []
        total_files = len(file_paths)
        
        try:
            logger.info(f"다중 파일 로딩 시작: {total_files} 파일")
            
            for i, file_path in enumerate(file_paths):
                try:
                    if progress_callback:
                        progress_callback(f"파일 {i+1}/{total_files} 처리 중: {Path(file_path).name}", 
                                        i / total_files)
                    
                    documents = self.load_documents(file_path)
                    all_documents.extend(documents)
                    
                except Exception as e:
                    error_msg = f"파일 처리 실패: {file_path} - {str(e)}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
            
            if progress_callback:
                progress_callback("다중 파일 로딩 완료", 1.0)
            
            logger.info(f"다중 파일 로딩 완료: {len(all_documents)} 총 문서")
            return all_documents
            
        except Exception as e:
            logger.error(f"다중 파일 로딩 중 오류: {str(e)}")
            return all_documents
    
    def _apply_postprocessing(self, documents: List[Document]) -> List[Document]:
        """후처리 적용"""
        try:
            # 추가적인 품질 검증
            validated_docs = []
            min_chunk_size = getattr(settings, 'min_chunk_size', 300)
            
            for doc in documents:
                content_length = len(doc.page_content.strip())
                
                # 특별한 경우: 문서가 작지만 의미있는 내용이 있으면 유지
                if content_length >= min_chunk_size or (content_length > 20 and content_length < min_chunk_size):
                    validated_docs.append(doc)
                    if content_length < min_chunk_size:
                        logger.info(f"짧은 문서 유지: {content_length}자 (min_chunk_size={min_chunk_size})")
            
            logger.info(f"후처리 검증: {len(documents)} -> {len(validated_docs)} 문서")
            return validated_docs
            
        except Exception as e:
            logger.error(f"후처리 중 오류: {str(e)}")
            return documents
    
    def _update_stats(self, file_path: str, documents_count: int, chunks_count: int, processing_time: float):
        """통계 정보 업데이트"""
        self.stats['files_processed'] += 1
        self.stats['documents_loaded'] += documents_count
        self.stats['chunks_created'] += chunks_count
        self.stats['processing_time'] += processing_time
    
    def get_supported_extensions(self) -> List[str]:
        """지원되는 파일 확장자 목록 반환"""
        return self.file_loader.get_supported_extensions()
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            'average_processing_time': (self.stats['processing_time'] / self.stats['files_processed'] 
                                       if self.stats['files_processed'] > 0 else 0),
            'average_chunks_per_file': (self.stats['chunks_created'] / self.stats['files_processed']
                                       if self.stats['files_processed'] > 0 else 0),
            'file_loader_stats': getattr(self.file_loader, 'get_stats', lambda: {})(),
            'text_processor_available': True,
            'document_splitter_stats': self.document_splitter.get_chunk_statistics([]) if hasattr(self.document_splitter, 'get_chunk_statistics') else {},
            'content_processor_stats': (self.content_processor.get_processing_statistics() 
                                       if self.content_processor else None)
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """현재 설정 정보 반환"""
        return {
            'use_ocr': self.use_ocr,
            'use_agent_preprocessing': self.use_agent_preprocessing,
            'enable_postprocessing': self.enable_postprocessing,
            'use_intelligent_image_extraction': self.use_intelligent_image_extraction,
            'preprocessing_model': self.preprocessing_model,
            'enable_multimodal_preprocessing': self.enable_multimodal_preprocessing,
            'supported_extensions': self.get_supported_extensions(),
            'modules': {
                'file_loader': type(self.file_loader).__name__,
                'text_processor': type(self.text_processor).__name__,
                'document_splitter': type(self.document_splitter).__name__,
                'content_processor': type(self.content_processor).__name__ if self.content_processor else None
            }
        }
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'files_processed': 0,
            'documents_loaded': 0,
            'chunks_created': 0,
            'processing_time': 0.0,
            'errors': []
        }
        logger.info("통계 정보 초기화 완료")


# 기존 API 호환성을 위한 별칭
EnhancedDocumentLoader = DocumentLoader


# 팩토리 함수
def create_document_loader(**kwargs) -> DocumentLoader:
    """
    DocumentLoader 인스턴스 생성 팩토리 함수
    
    Args:
        **kwargs: DocumentLoader 초기화 매개변수
        
    Returns:
        DocumentLoader 인스턴스
    """
    return DocumentLoader(**kwargs)


def create_simple_loader() -> DocumentLoader:
    """
    기본 설정의 간단한 로더 생성
    
    Returns:
        기본 설정의 DocumentLoader 인스턴스
    """
    return DocumentLoader(
        use_ocr=False,
        use_agent_preprocessing=False,
        enable_postprocessing=False,
        use_intelligent_image_extraction=False,
        enable_multimodal_preprocessing=False
    )


def create_advanced_loader() -> DocumentLoader:
    """
    고급 기능이 활성화된 로더 생성
    
    Returns:
        고급 기능 활성화된 DocumentLoader 인스턴스
    """
    return DocumentLoader(
        use_ocr=True,
        use_agent_preprocessing=True,
        enable_postprocessing=True,
        use_intelligent_image_extraction=True,
        enable_multimodal_preprocessing=True
    )
