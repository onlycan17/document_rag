"""
문서 분할 모듈

이 모듈은 기존 document_loader.py에서 문서 분할 및 청킹 전략 로직을 분리하여
단일 책임 원칙에 따라 독립적으로 관리합니다.

주요 기능:
- 다양한 청킹 전략 지원
- 의미 기반 분할
- 크기 기반 분할
- 구조 기반 분할
- 청크 품질 최적화
- 중복 및 병합 처리
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)

# 설정값 import
try:
    from config import settings
except ImportError:
    # 설정이 없는 경우 기본값 사용
    class DefaultSettings:
        chunk_size = 1200
        chunk_overlap = 200
        use_semantic_chunking = True
        min_chunk_size = 300
        max_chunk_size = 2000
        merge_threshold = 500
    
    settings = DefaultSettings()

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """
    문서 분할 및 청킹 클래스
    
    기존 document_loader.py에서 문서 분할 로직을 분리하여
    다양한 분할 전략을 지원하는 모듈로 구성
    """
    
    def __init__(self, 
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 min_chunk_size: Optional[int] = None,
                 max_chunk_size: Optional[int] = None):
        """
        DocumentSplitter 초기화
        
        Args:
            chunk_size: 기본 청크 크기
            chunk_overlap: 청크 간 중복 크기
            min_chunk_size: 최소 청크 크기
            max_chunk_size: 최대 청크 크기
        """
        # 설정값 초기화
        self.chunk_size = chunk_size or getattr(settings, 'chunk_size', 1200)
        self.chunk_overlap = chunk_overlap or getattr(settings, 'chunk_overlap', 200)
        self.min_chunk_size = min_chunk_size or getattr(settings, 'min_chunk_size', 300)
        self.max_chunk_size = max_chunk_size or getattr(settings, 'max_chunk_size', 2000)
        self.merge_threshold = getattr(settings, 'merge_threshold', 500)
        self.use_semantic_chunking = getattr(settings, 'use_semantic_chunking', True)
        
        # 다양한 스플리터 초기화
        self.splitters = self._init_splitters()
        
        # 분할 패턴 초기화
        self.split_patterns = self._init_split_patterns()
        
        # 병합 규칙 초기화
        self.merge_rules = self._init_merge_rules()
        
        logger.info(f"DocumentSplitter 초기화 완료 - 청크 크기: {self.chunk_size}, 중복: {self.chunk_overlap}")
    
    def _init_splitters(self) -> Dict[str, Any]:
        """다양한 텍스트 스플리터 초기화"""
        splitters = {}
        
        try:
            # 기본 재귀 분할기
            splitters['recursive'] = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # 문단 구분
                    "\n",    # 줄 구분
                    ".",     # 문장 구분
                    "!",     # 문장 구분
                    "?",     # 문장 구분
                    ";",     # 절 구분
                    ",",     # 구문 구분
                    " ",     # 단어 구분
                    ""       # 문자 구분
                ]
            )
            
            # 문자 기반 분할기
            splitters['character'] = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n"
            )
            
            # 마크다운 헤더 기반 분할기
            splitters['markdown'] = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ]
            )
            
            # 한국어 최적화 분할기
            splitters['korean'] = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",     # 문단 구분
                    "\n",       # 줄 구분
                    ".",        # 문장 구분 (한국어)
                    "!",        # 감탄문
                    "?",        # 의문문
                    "。",       # 일본식 마침표
                    "다.",      # 한국어 서술문 종결
                    "요.",      # 한국어 존댓말 종결
                    "니다.",    # 한국어 격식체 종결
                    "습니다.",  # 한국어 격식체 종결
                    ";",        # 절 구분
                    ":",        # 설명 구분
                    ",",        # 구문 구분
                    " ",        # 단어 구분
                    ""          # 문자 구분
                ]
            )
            
        except Exception as e:
            logger.error(f"스플리터 초기화 중 오류: {str(e)}")
            # 기본 스플리터라도 생성
            splitters['recursive'] = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        return splitters
    
    def _init_split_patterns(self) -> Dict[str, Any]:
        """분할 패턴 초기화"""
        return {
            # 문단 분할 패턴
            'paragraph': re.compile(r'\n\s*\n'),
            
            # 문장 분할 패턴 (한국어 고려)
            'sentence': re.compile(r'[.!?。][\s\n]+|다\.[\s\n]+|요\.[\s\n]+|니다\.[\s\n]+|습니다\.[\s\n]+'),
            
            # 섹션 분할 패턴
            'section': re.compile(r'^(제?\s*\d+\s*[장절편부].*?)$|^(#{1,6}\s*.+)$', re.MULTILINE),
            
            # 목록 분할 패턴
            'list': re.compile(r'^\s*[•·▪▫\-\*\+]\s+|^\s*\d+\.\s+', re.MULTILINE),
            
            # 테이블 분할 패턴
            'table': re.compile(r'^\|.*\|$', re.MULTILINE),
            
            # 코드 블록 패턴
            'code': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            
            # 인용구 패턴
            'quote': re.compile(r'^>\s+', re.MULTILINE)
        }
    
    def _init_merge_rules(self) -> Dict[str, Callable]:
        """청크 병합 규칙 초기화"""
        return {
            'size_based': self._should_merge_by_size,
            'semantic_based': self._should_merge_semantically,
            'structure_based': self._should_merge_by_structure
        }
    
    def split_documents(self, documents: List[Document], 
                       strategy: str = 'auto',
                       preserve_structure: bool = True) -> List[Document]:
        """
        문서 리스트 분할
        
        Args:
            documents: 분할할 문서 리스트
            strategy: 분할 전략 ('auto', 'recursive', 'semantic', 'markdown', 'korean')
            preserve_structure: 구조 보존 여부
            
        Returns:
            분할된 문서 리스트
        """
        if not documents:
            return []
        
        try:
            all_chunks = []
            
            for doc in documents:
                if not doc.page_content or not doc.page_content.strip():
                    continue
                
                # 문서별 최적 전략 결정
                if strategy == 'auto':
                    chosen_strategy = self._choose_strategy(doc.page_content)
                else:
                    chosen_strategy = strategy
                
                # 문서 분할
                chunks = self._split_single_document(
                    doc, 
                    chosen_strategy, 
                    preserve_structure
                )
                
                all_chunks.extend(chunks)
            
            # 청크 품질 개선
            optimized_chunks = self._optimize_chunks(all_chunks)
            
            logger.info(f"문서 분할 완료: {len(documents)} 문서 -> {len(optimized_chunks)} 청크")
            return optimized_chunks
            
        except Exception as e:
            logger.error(f"문서 분할 중 오류: {str(e)}")
            return documents
    
    def _choose_strategy(self, text: str) -> str:
        """텍스트 특성에 따른 최적 분할 전략 선택"""
        try:
            # 한국어 비율 계산
            korean_chars = len(re.findall(r'[가-힣]', text))
            total_chars = len(text.replace(' ', ''))
            korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
            
            # 마크다운 패턴 검사
            has_markdown = bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE) or 
                              '```' in text or 
                              re.search(r'^\|.*\|$', text, re.MULTILINE))
            
            # 구조화된 텍스트 검사
            has_structure = bool(re.search(r'^제?\s*\d+\s*[장절편부]', text, re.MULTILINE) or
                               re.search(r'^\s*\d+\.\s+', text, re.MULTILINE))
            
            # 전략 선택
            if has_markdown:
                return 'markdown'
            elif korean_ratio > 0.3:
                return 'korean'
            elif has_structure:
                return 'semantic'
            else:
                return 'recursive'
                
        except Exception as e:
            logger.error(f"전략 선택 중 오류: {str(e)}")
            return 'recursive'
    
    def _split_single_document(self, document: Document, 
                              strategy: str, 
                              preserve_structure: bool) -> List[Document]:
        """단일 문서 분할"""
        try:
            text = document.page_content
            metadata = document.metadata.copy() if document.metadata else {}
            
            # 전략별 분할 실행
            if strategy == 'semantic':
                chunks = self._semantic_split(text, metadata)
            elif strategy == 'markdown':
                chunks = self._markdown_split(text, metadata)
            elif strategy == 'korean':
                chunks = self._korean_split(text, metadata)
            else:
                chunks = self._recursive_split(text, metadata)
            
            # 구조 정보 추가
            if preserve_structure:
                chunks = self._add_structure_info(chunks, text)
            
            # 청크 메타데이터 보강
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'split_strategy': strategy,
                    'original_length': len(text),
                    'chunk_length': len(chunk.page_content)
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"문서 분할 중 오류: {str(e)}")
            return [document]
    
    def _semantic_split(self, text: str, metadata: Dict) -> List[Document]:
        """의미 기반 분할"""
        try:
            chunks = []
            
            # 섹션으로 먼저 분할
            sections = self._split_by_sections(text)
            
            for section in sections:
                if len(section) <= self.max_chunk_size:
                    # 섹션이 충분히 작으면 그대로 사용
                    if len(section.strip()) >= self.min_chunk_size:
                        chunks.append(Document(
                            page_content=section.strip(),
                            metadata=metadata.copy()
                        ))
                else:
                    # 섹션이 크면 문단으로 추가 분할
                    paragraphs = self._split_by_paragraphs(section)
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk + paragraph) <= self.chunk_size:
                            current_chunk += paragraph + "\n\n"
                        else:
                            if current_chunk.strip():
                                chunks.append(Document(
                                    page_content=current_chunk.strip(),
                                    metadata=metadata.copy()
                                ))
                            current_chunk = paragraph + "\n\n"
                    
                    if current_chunk.strip():
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata=metadata.copy()
                        ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"의미 기반 분할 중 오류: {str(e)}")
            return self._recursive_split(text, metadata)
    
    def _markdown_split(self, text: str, metadata: Dict) -> List[Document]:
        """마크다운 기반 분할"""
        try:
            splitter = self.splitters['markdown']
            md_docs = splitter.split_text(text)
            
            chunks = []
            for md_doc in md_docs:
                if isinstance(md_doc, str):
                    content = md_doc
                    doc_metadata = metadata.copy()
                else:
                    content = md_doc.page_content
                    doc_metadata = metadata.copy()
                    doc_metadata.update(md_doc.metadata)
                
                # 크기가 너무 크면 추가 분할
                if len(content) > self.max_chunk_size:
                    sub_chunks = self._recursive_split_text(content)
                    for sub_content in sub_chunks:
                        if len(sub_content.strip()) >= self.min_chunk_size:
                            chunks.append(Document(
                                page_content=sub_content.strip(),
                                metadata=doc_metadata.copy()
                            ))
                else:
                    if len(content.strip()) >= self.min_chunk_size:
                        chunks.append(Document(
                            page_content=content.strip(),
                            metadata=doc_metadata
                        ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"마크다운 분할 중 오류: {str(e)}")
            return self._recursive_split(text, metadata)
    
    def _korean_split(self, text: str, metadata: Dict) -> List[Document]:
        """한국어 최적화 분할"""
        try:
            splitter = self.splitters['korean']
            korean_chunks = splitter.split_text(text)
            
            chunks = []
            for content in korean_chunks:
                if len(content.strip()) >= self.min_chunk_size:
                    chunks.append(Document(
                        page_content=content.strip(),
                        metadata=metadata.copy()
                    ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"한국어 분할 중 오류: {str(e)}")
            return self._recursive_split(text, metadata)
    
    def _recursive_split(self, text: str, metadata: Dict) -> List[Document]:
        """재귀적 분할"""
        try:
            # 특별한 경우: 원본 텍스트가 매우 작으면 그대로 유지
            if len(text.strip()) < self.min_chunk_size and len(text.strip()) > 20:
                logger.info(f"원본 텍스트가 작음 ({len(text.strip())}자), 분할 없이 그대로 반환")
                return [Document(page_content=text.strip(), metadata=metadata.copy())]
            
            splitter = self.splitters['recursive']
            recursive_chunks = splitter.split_text(text)
            
            chunks = []
            for content in recursive_chunks:
                if len(content.strip()) >= self.min_chunk_size:
                    chunks.append(Document(
                        page_content=content.strip(),
                        metadata=metadata.copy()
                    ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"재귀적 분할 중 오류: {str(e)}")
            return [Document(page_content=text, metadata=metadata)]
    
    def _recursive_split_text(self, text: str) -> List[str]:
        """텍스트만 재귀적 분할"""
        try:
            splitter = self.splitters['recursive']
            return splitter.split_text(text)
        except Exception as e:
            logger.error(f"텍스트 재귀적 분할 중 오류: {str(e)}")
            return [text]
    
    def _split_by_sections(self, text: str) -> List[str]:
        """섹션별 분할"""
        try:
            sections = []
            lines = text.split('\n')
            current_section = []
            
            for line in lines:
                # 섹션 헤더 검사
                if re.match(r'^(제?\s*\d+\s*[장절편부].*?)$|^(#{1,6}\s*.+)$', line.strip()):
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                
                current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            return [section for section in sections if section.strip()]
            
        except Exception as e:
            logger.error(f"섹션 분할 중 오류: {str(e)}")
            return [text]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """문단별 분할"""
        try:
            paragraphs = self.split_patterns['paragraph'].split(text)
            return [p.strip() for p in paragraphs if p.strip()]
        except Exception as e:
            logger.error(f"문단 분할 중 오류: {str(e)}")
            return [text]
    
    def _add_structure_info(self, chunks: List[Document], original_text: str) -> List[Document]:
        """청크에 구조 정보 추가"""
        try:
            for i, chunk in enumerate(chunks):
                # 청크의 원본 텍스트 내 위치 찾기
                start_pos = original_text.find(chunk.page_content[:50])
                if start_pos != -1:
                    chunk.metadata['text_position'] = start_pos
                
                # 주변 컨텍스트 정보
                if i > 0:
                    chunk.metadata['previous_chunk'] = chunks[i-1].page_content[:100]
                if i < len(chunks) - 1:
                    chunk.metadata['next_chunk'] = chunks[i+1].page_content[:100]
                
                # 구조적 특성
                content = chunk.page_content
                chunk.metadata['has_headers'] = bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE))
                chunk.metadata['has_lists'] = bool(re.search(r'^\s*[-*+\d]+[\.\)]\s+', content, re.MULTILINE))
                chunk.metadata['has_tables'] = bool(re.search(r'^\|.*\|$', content, re.MULTILINE))
                chunk.metadata['has_code'] = bool('```' in content or '`' in content)
            
            return chunks
            
        except Exception as e:
            logger.error(f"구조 정보 추가 중 오류: {str(e)}")
            return chunks
    
    def _optimize_chunks(self, chunks: List[Document]) -> List[Document]:
        """청크 품질 최적화"""
        try:
            # 1단계: 너무 작은 청크 병합
            merged_chunks = self._merge_small_chunks(chunks)
            
            # 2단계: 중복 제거
            deduplicated_chunks = self._remove_duplicate_chunks(merged_chunks)
            
            # 3단계: 청크 품질 검증
            validated_chunks = self._validate_chunk_quality(deduplicated_chunks)
            
            logger.info(f"청크 최적화: {len(chunks)} -> {len(validated_chunks)} 청크")
            return validated_chunks
            
        except Exception as e:
            logger.error(f"청크 최적화 중 오류: {str(e)}")
            return chunks
    
    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """작은 청크들 병합"""
        if not chunks:
            return []
        
        try:
            merged = []
            i = 0
            
            while i < len(chunks):
                current = chunks[i]
                
                # 현재 청크가 작고 다음 청크와 병합 가능한지 확인
                if (len(current.page_content) < self.merge_threshold and 
                    i < len(chunks) - 1):
                    
                    next_chunk = chunks[i + 1]
                    combined_length = len(current.page_content) + len(next_chunk.page_content)
                    
                    # 병합 가능한 크기인지 확인
                    if combined_length <= self.chunk_size * 1.5:  # 50% 여유
                        # 병합 실행
                        merged_content = current.page_content + "\n\n" + next_chunk.page_content
                        merged_metadata = current.metadata.copy()
                        merged_metadata.update({
                            'merged_chunks': 2,
                            'original_indices': [
                                current.metadata.get('chunk_index', i),
                                next_chunk.metadata.get('chunk_index', i + 1)
                            ]
                        })
                        
                        merged.append(Document(
                            page_content=merged_content,
                            metadata=merged_metadata
                        ))
                        
                        i += 2  # 두 청크를 건너뛰기
                        continue
                
                merged.append(current)
                i += 1
            
            return merged
            
        except Exception as e:
            logger.error(f"작은 청크 병합 중 오류: {str(e)}")
            return chunks
    
    def _remove_duplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """중복 청크 제거"""
        if not chunks:
            return []
        
        try:
            unique_chunks = []
            seen_contents = set()
            
            for chunk in chunks:
                # 내용의 정규화된 해시 생성
                normalized_content = re.sub(r'\s+', ' ', chunk.page_content.strip().lower())
                content_hash = hash(normalized_content)
                
                if content_hash not in seen_contents:
                    unique_chunks.append(chunk)
                    seen_contents.add(content_hash)
            
            return unique_chunks
            
        except Exception as e:
            logger.error(f"중복 청크 제거 중 오류: {str(e)}")
            return chunks
    
    def _validate_chunk_quality(self, chunks: List[Document]) -> List[Document]:
        """청크 품질 검증"""
        try:
            valid_chunks = []
            
            # 특별한 경우: 원본 문서 자체가 매우 작은 경우 예외 처리
            if len(chunks) == 1 and chunks[0].page_content.strip():
                original_content = chunks[0].page_content.strip()
                if len(original_content) < self.min_chunk_size and len(original_content) > 20:
                    logger.info(f"원본 문서가 작음 ({len(original_content)}자), 최소 크기 제한 무시")
                    valid_chunks.append(chunks[0])
                    return valid_chunks
            
            for chunk in chunks:
                content = chunk.page_content.strip()
                
                # 기본 품질 검증
                if not content:
                    continue
                
                if len(content) < self.min_chunk_size:
                    continue
                
                if len(content) > self.max_chunk_size:
                    # 너무 큰 청크는 추가 분할
                    sub_chunks = self._recursive_split_text(content)
                    for sub_content in sub_chunks:
                        if len(sub_content.strip()) >= self.min_chunk_size:
                            sub_metadata = chunk.metadata.copy()
                            sub_metadata['split_from_large'] = True
                            valid_chunks.append(Document(
                                page_content=sub_content.strip(),
                                metadata=sub_metadata
                            ))
                    continue
                
                # 내용 품질 검증
                if self._is_meaningful_content(content):
                    valid_chunks.append(chunk)
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"청크 품질 검증 중 오류: {str(e)}")
            return chunks
    
    def _is_meaningful_content(self, content: str) -> bool:
        """의미있는 내용인지 검증"""
        try:
            # 너무 많은 반복 문자 검사
            if re.search(r'(.)\1{10,}', content):
                return False
            
            # 의미있는 단어 비율 검사
            words = content.split()
            if len(words) < 5:
                return False
            
            # 한국어나 영어 단어가 포함되어 있는지 검사
            meaningful_words = len([w for w in words if re.search(r'[가-힣a-zA-Z]{2,}', w)])
            if meaningful_words / len(words) < 0.3:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"의미 검증 중 오류: {str(e)}")
            return True  # 오류 시 보수적으로 유지
    
    def _should_merge_by_size(self, chunk1: Document, chunk2: Document) -> bool:
        """크기 기반 병합 가능성 검사"""
        combined_size = len(chunk1.page_content) + len(chunk2.page_content)
        return combined_size <= self.chunk_size
    
    def _should_merge_semantically(self, chunk1: Document, chunk2: Document) -> bool:
        """의미 기반 병합 가능성 검사 (간단한 버전)"""
        # 간단한 키워드 유사도 검사
        words1 = set(chunk1.page_content.lower().split())
        words2 = set(chunk2.page_content.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Jaccard 유사도
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union > 0.3
    
    def _should_merge_by_structure(self, chunk1: Document, chunk2: Document) -> bool:
        """구조 기반 병합 가능성 검사"""
        # 둘 다 같은 종류의 구조적 요소인지 검사
        meta1 = chunk1.metadata
        meta2 = chunk2.metadata
        
        # 연속된 목록 항목인지 검사
        if (meta1.get('has_lists') and meta2.get('has_lists')):
            return True
        
        # 같은 섹션의 연속된 청크인지 검사
        if (abs(meta1.get('chunk_index', 0) - meta2.get('chunk_index', 0)) == 1):
            return True
        
        return False
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """청크 통계 정보 생성"""
        if not chunks:
            return {}
        
        try:
            sizes = [len(chunk.page_content) for chunk in chunks]
            
            stats = {
                "total_chunks": len(chunks),
                "total_characters": sum(sizes),
                "average_size": sum(sizes) / len(sizes),
                "min_size": min(sizes),
                "max_size": max(sizes),
                "sizes": sizes
            }
            
            # 크기 분포
            small_chunks = len([s for s in sizes if s < self.min_chunk_size])
            large_chunks = len([s for s in sizes if s > self.max_chunk_size])
            
            stats.update({
                "small_chunks": small_chunks,
                "large_chunks": large_chunks,
                "optimal_chunks": len(chunks) - small_chunks - large_chunks,
                "quality_ratio": (len(chunks) - small_chunks - large_chunks) / len(chunks)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"통계 생성 중 오류: {str(e)}")
            return {}