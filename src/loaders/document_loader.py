from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from config import settings
from .pdf_loader_advanced import AdvancedPDFLoader
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, use_ocr: bool = True):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "。", "!", "?", ";", "；", ",", "，", " ", ""]
        )
        self.use_ocr = use_ocr
        self.advanced_pdf_loader = AdvancedPDFLoader(use_ocr=use_ocr)
        
        # OCR 사용 가능 여부 확인
        if use_ocr:
            ocr_available = AdvancedPDFLoader.check_ocr_availability()
            if not ocr_available:
                logger.warning("OCR을 사용할 수 없습니다. Tesseract와 한국어 언어팩을 설치해주세요.")
                self.use_ocr = False
        
    def load_document(self, file_path: str, progress_callback=None) -> List[Document]:
        """단일 문서를 로드하고 청크로 분할"""
        file_extension = Path(file_path).suffix.lower()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if progress_callback:
            progress_callback(0.1, f"파일 로드 중... ({file_size_mb:.1f}MB)")
        
        if file_extension == '.txt' or file_extension == '.md':
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            if progress_callback:
                progress_callback(0.5, "텍스트 추출 완료")
        elif file_extension == '.pdf':
            if self.use_ocr:
                try:
                    # 고급 PDF 로더 사용 (OCR 포함)
                    if progress_callback:
                        progress_callback(0.2, "PDF 분석 중... (OCR 모드)")
                    documents = self.advanced_pdf_loader.load_pdf(file_path, progress_callback)
                    logger.info(f"고급 PDF 로더로 처리: {file_path}")
                except Exception as e:
                    logger.warning(f"고급 PDF 로더 실패, 기본 로더 사용: {str(e)}")
                    if progress_callback:
                        progress_callback(0.3, "기본 PDF 로더로 전환...")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
            else:
                if progress_callback:
                    progress_callback(0.2, "PDF 텍스트 추출 중...")
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if progress_callback:
                    progress_callback(0.5, "텍스트 추출 완료")
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
        
        if progress_callback:
            progress_callback(0.7, f"문서 분할 중... ({len(documents)}개 페이지)")
        
        chunks = self.text_splitter.split_documents(documents)
        
        if progress_callback:
            progress_callback(0.9, f"분할 완료! ({len(chunks)}개 청크)")
        
        return chunks
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """디렉토리 내의 모든 문서를 로드하고 청크로 분할"""
        all_documents = []
        supported_extensions = ['.txt', '.md', '.pdf']
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        documents = self.load_document(file_path)
                        # 메타데이터 추가
                        for doc in documents:
                            doc.metadata.update({
                                'source': file_path,
                                'file_name': file,
                                'file_type': Path(file).suffix.lower()
                            })
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
                'id': f"doc_{i}",
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            processed_docs.append(processed_doc)
        
        return processed_docs