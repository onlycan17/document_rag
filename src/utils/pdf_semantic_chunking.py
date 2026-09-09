#!/usr/bin/env python3
"""
PDF 의미 기반 청킹 믹스인

ImprovedPDFConverter에서 마크다운 → 청크 분할 책임만 분리한 믹스인.
인스턴스 속성(enable_semantic_chunking, semantic_chunker)은 본체에서 제공된다.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class SemanticChunkingMixin:
    """마크다운 콘텐츠 의미 기반 청킹 기능"""

    def create_semantic_chunks(self, markdown_content: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        마크다운 내용을 의미 기반으로 청크 분할

        Args:
            markdown_content: 분할할 마크다운 텍스트
            metadata: 추가 메타데이터

        Returns:
            List[Dict]: 의미 기반 청크 리스트
        """
        if not self.enable_semantic_chunking or not self.semantic_chunker:
            # 의미 기반 청킹을 사용할 수 없는 경우 기본 청킹
            return self._create_basic_chunks(markdown_content, metadata)

        logger.info("의미 기반 청킹 시작")
        chunks = self.semantic_chunker.create_semantic_chunks(markdown_content, metadata)

        # 청킹 통계 로깅
        if chunks:
            stats = self.semantic_chunker.get_chunk_statistics(chunks)
            logger.info(
                f"의미 기반 청킹 완료: {stats['total_chunks']}개 청크, "
                f"평균 크기: {stats['avg_chunk_size']:.0f}자, "
                f"평균 일관성: {stats['avg_coherence']:.2f}, "
                f"평균 도메인 관련성: {stats['avg_domain_relevance']:.2f}"
            )

        return chunks

    def _create_basic_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """기본 청킹 방식 (의미 기반 청킹을 사용할 수 없는 경우)"""
        if not text.strip():
            return []

        chunks = []
        max_chunk_size = 1200
        min_chunk_size = 300

        # 문단별로 분할
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            # 현재 청크에 문단을 추가했을 때 크기 확인
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph

            if len(potential_chunk) > max_chunk_size and current_chunk:
                # 현재 청크가 최소 크기를 만족하면 저장
                if len(current_chunk) >= min_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update(
                        {"chunk_method": "basic", "chunk_id": chunk_id, "chunk_size": len(current_chunk)}
                    )

                    chunks.append(
                        {
                            "text": current_chunk,
                            "metadata": chunk_metadata,
                            "semantic_info": {"chunk_id": chunk_id, "keywords": []},
                        }
                    )

                    chunk_id += 1
                    current_chunk = paragraph
                else:
                    current_chunk = potential_chunk
            else:
                current_chunk = potential_chunk

        # 마지막 청크 처리
        if current_chunk:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({"chunk_method": "basic", "chunk_id": chunk_id, "chunk_size": len(current_chunk)})

            chunks.append(
                {
                    "text": current_chunk,
                    "metadata": chunk_metadata,
                    "semantic_info": {"chunk_id": chunk_id, "keywords": []},
                }
            )

        return chunks

    def convert_pdf_to_semantic_chunks(
        self, pdf_path: str, progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Dict], int]:
        """
        PDF를 의미 기반 청크로 변환

        Args:
            pdf_path: PDF 파일 경로
            progress_callback: 진행 상황 콜백 함수

        Returns:
            Tuple[청크 리스트, 이미지 개수]
        """
        # 1단계: 일반 마크다운 변환
        markdown_content, image_count = self.convert_pdf_to_markdown(pdf_path, progress_callback)

        if not markdown_content:
            return [], image_count

        # 2단계: 의미 기반 청킹 적용
        if progress_callback:
            progress_callback(0.95, "의미 기반 청킹 처리 중...")

        metadata = {
            "source_file": Path(pdf_path).name,
            "conversion_date": datetime.now().isoformat(),
            "image_count": image_count,
            "processing_method": "improved_semantic",
        }

        chunks = self.create_semantic_chunks(markdown_content, metadata)

        if progress_callback:
            progress_callback(1.0, f"의미 기반 청킹 완료: {len(chunks)}개 청크 생성")

        return chunks, image_count
