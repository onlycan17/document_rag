from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import os
import re
from pathlib import Path
from config import settings
from .pdf_loader_advanced import AdvancedPDFLoader
import logging

logger = logging.getLogger(__name__)

class EnhancedDocumentLoader:
    """
    향상된 문서 로더 클래스
    - 의미 기반 청킹 지원
    - 한국어 문서 최적화
    - 다양한 청킹 전략 지원
    """
    
    def __init__(self, use_ocr: bool = True):
        # 기본 텍스트 분할기 (기존 방식)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "。", "!", "?", ";", "；", ",", "，", " ", ""]
        )
        
        # 의미 기반 분할기 (향상된 방식)
        if settings.use_semantic_chunking:
            try:
                self.semantic_splitter = SentenceTransformersTokenTextSplitter(
                    chunk_overlap=settings.chunk_overlap,
                    model_name=getattr(settings, 'korean_embedding_model', settings.embedding_model_name),
                    tokens_per_chunk=settings.chunk_size
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
    
    def load_document(self, file_path: str, progress_callback=None) -> List[Document]:
        """
        단일 문서를 로드하고 최적화된 청크로 분할
        - 파일 유형별 최적화된 로딩
        - 향상된 전처리
        - 의미 기반 청킹 지원
        """
        file_extension = Path(file_path).suffix.lower()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if progress_callback:
            progress_callback(0.1, f"파일 로드 중... ({file_size_mb:.1f}MB)")
        
        # 1. 파일 유형별 로딩
        if file_extension in ['.txt', '.md']:
            documents = self._load_text_file(file_path, progress_callback)
        elif file_extension == '.pdf':
            documents = self._load_pdf_file(file_path, progress_callback)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
        
        if progress_callback:
            progress_callback(0.6, f"문서 전처리 중... ({len(documents)}개 페이지)")
        
        # 2. 문서 전처리 및 정제
        processed_documents = self._preprocess_documents(documents)
        
        if progress_callback:
            progress_callback(0.7, f"문서 분할 중... (전처리 완료)")
        
        # 3. 청킹 전략에 따른 분할
        chunks = self._split_documents_optimized(processed_documents)
        
        # 4. 메타데이터 보강
        enhanced_chunks = self._enhance_metadata(chunks, file_path)
        
        if progress_callback:
            progress_callback(0.9, f"분할 완료! ({len(enhanced_chunks)}개 청크)")
        
        logger.info(f"문서 로딩 완료: {file_path} -> {len(enhanced_chunks)}개 청크")
        return enhanced_chunks
    
    def _load_text_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """텍스트 파일 로딩 최적화"""
        try:
            # 다양한 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"파일 인코딩 감지: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise ValueError("지원되는 인코딩을 찾을 수 없습니다")
            
            if progress_callback:
                progress_callback(0.5, "텍스트 추출 완료")
            
            return [Document(page_content=content, metadata={'source': file_path})]
            
        except Exception as e:
            logger.error(f"텍스트 파일 로딩 실패: {str(e)}")
            # 기본 로더로 폴백
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
    
    def _load_pdf_file(self, file_path: str, progress_callback=None) -> List[Document]:
        """PDF 파일 로딩 최적화"""
        if self.use_ocr:
            try:
                if progress_callback:
                    progress_callback(0.2, "PDF 분석 중... (OCR 모드)")
                documents = self.advanced_pdf_loader.load_pdf(file_path, progress_callback)
                logger.info(f"고급 PDF 로더로 처리: {file_path}")
                return documents
            except Exception as e:
                logger.warning(f"고급 PDF 로더 실패, 기본 로더 사용: {str(e)}")
                if progress_callback:
                    progress_callback(0.3, "기본 PDF 로더로 전환...")
        
        # 기본 PDF 로더
        if progress_callback:
            progress_callback(0.2, "PDF 텍스트 추출 중...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if progress_callback:
            progress_callback(0.5, "텍스트 추출 완료")
        
        return documents
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 전처리 및 정제
        - 불필요한 내용 제거
        - 텍스트 정규화
        - 한국어 최적화
        """
        processed_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # 1. 기본 정제
            content = self._clean_text(content)
            
            # 2. 한국어 특화 정제
            content = self._korean_text_normalization(content)
            
            # 3. 구조화된 내용 보존
            content = self._preserve_structure(content)
            
            # 4. 너무 짧은 내용 필터링 강화 (300자 이상)
            if len(content.strip()) >= 300:  # 최소 길이를 300자로 증가
                processed_docs.append(Document(
                    page_content=content,
                    metadata=doc.metadata
                ))
            else:
                logger.debug(f"너무 짧은 콘텐츠 필터링: {len(content)}자 - {content[:50]}...")
        
        return processed_docs
    
    def _clean_text(self, text: str) -> str:
        """기본 텍스트 정제"""
        if not text:
            return ""
        
        # 연속된 공백 및 줄바꿈 정리
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # 3개 이상의 연속 줄바꿈을 2개로
        text = re.sub(r' +', ' ', text)  # 연속된 공백을 하나로
        
        # 특수 문자 정리
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # BOM
        text = text.replace('\xa0', ' ')   # Non-breaking space
        text = text.replace('\u3000', ' ') # Ideographic space
        
        # 이상한 문자 제거 (제어 문자 등)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text.strip()
    
    def _korean_text_normalization(self, text: str) -> str:
        """한국어 텍스트 정규화"""
        # 한글 자모 결합 문제 해결
        text = re.sub(r'([ㄱ-ㅎ])([ㅏ-ㅣ])', r'\1\2', text)
        
        # 한국어 문장 부호 정규화
        text = text.replace('．', '.')
        text = text.replace('，', ',')
        text = text.replace('；', ';')
        text = text.replace('：', ':')
        
        # 반복되는 특수문자 제거
        text = re.sub(r'[─]{2,}', '─', text)
        text = re.sub(r'[=]{3,}', '===', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text
    
    def _preserve_structure(self, text: str) -> str:
        """문서 구조 보존 (제목, 목록 등)"""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue
            
            # 제목 형태 보존 (번호가 있는 제목)
            if re.match(r'^\d+\.?\s+[가-힣]', line):
                processed_lines.append(f"\n{line}\n")
            # 목록 항목 보존
            elif re.match(r'^[•▪▫-]\s+', line):
                processed_lines.append(line)
            # 일반 텍스트
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _split_documents_optimized(self, documents: List[Document]) -> List[Document]:
        """
        최적화된 문서 분할
        - 설정에 따라 의미 기반 또는 일반 분할 선택
        - 한국어 문장 구조 고려
        """
        if settings.use_semantic_chunking and self.semantic_splitter:
            logger.info("의미 기반 청킹 사용")
            try:
                chunks = self.semantic_splitter.split_documents(documents)
                # 의미 기반 청킹 성공 시 후처리
                return self._post_process_semantic_chunks(chunks)
            except Exception as e:
                logger.warning(f"의미 기반 청킹 실패, 기본 청킹 사용: {str(e)}")
        
        # 기본 청킹 (향상된 버전)
        logger.info("향상된 기본 청킹 사용")
        return self._enhanced_default_chunking(documents)
    
    def _enhanced_default_chunking(self, documents: List[Document]) -> List[Document]:
        """향상된 기본 청킹"""
        # 한국어에 최적화된 분리자 순서
        korean_optimized_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # 여러 줄바꿈
                "\n\n",    # 문단 분리
                "\n",      # 줄바꿈
                "。",      # 한국어 마침표 (문서에 따라)
                ".",       # 영어 마침표
                "!",       # 느낌표
                "?",       # 물음표
                ";",       # 세미콜론
                ":",       # 콜론
                ",",       # 쉼표
                " ",       # 공백
                ""         # 문자 단위
            ]
        )
        
        chunks = korean_optimized_splitter.split_documents(documents)
        
        # 기본 청킹에도 후처리 적용
        return self._post_process_semantic_chunks(chunks)
    
    def _post_process_semantic_chunks(self, chunks: List[Document]) -> List[Document]:
        """의미 기반 청킹 후처리 - 짧은 청크 병합 강화"""
        processed_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            
            # 너무 짧은 청크는 이전 청크와 합치기 (500자 미만)
            if len(content) < 500 and processed_chunks:
                last_chunk = processed_chunks[-1]
                combined_content = last_chunk.page_content + "\n\n" + content
                
                # 합쳐도 최대 크기를 넘지 않으면 합치기
                if len(combined_content) <= settings.chunk_size * 1.5:  # 1.2에서 1.5로 증가
                    processed_chunks[-1] = Document(
                        page_content=combined_content,
                        metadata=last_chunk.metadata
                    )
                    logger.debug(f"짧은 청크 병합: {len(content)}자 -> {len(combined_content)}자")
                    continue
            
            # 여전히 너무 짧은 청크는 제외
            if len(content) >= 300:  # 최소 크기 보장
                processed_chunks.append(chunk)
            else:
                logger.debug(f"너무 짧은 청크 제외: {len(content)}자 - {content[:50]}...")
        
        return processed_chunks
    
    def _enhance_metadata(self, chunks: List[Document], file_path: str) -> List[Document]:
        """메타데이터 보강"""
        enhanced_chunks = []
        file_name = Path(file_path).name
        
        for i, chunk in enumerate(chunks):
            # 기존 메타데이터 복사
            metadata = chunk.metadata.copy()
            
            # 추가 메타데이터
            metadata.update({
                'source': file_path,
                'file_name': file_name,
                'file_type': Path(file_path).suffix.lower(),
                'chunk_id': f"{file_name}_{i:04d}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk.page_content),
                'processing_method': 'semantic' if settings.use_semantic_chunking and self.semantic_splitter else 'enhanced_default'
            })
            
            enhanced_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=metadata
            ))
        
        return enhanced_chunks
    
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
                'id': doc.metadata.get('chunk_id', f"doc_{i}"),
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            processed_docs.append(processed_doc)
        
        return processed_docs

# 기존 DocumentLoader와의 호환성 유지
class DocumentLoader(EnhancedDocumentLoader):
    """기존 DocumentLoader와의 호환성을 위한 클래스"""
    pass