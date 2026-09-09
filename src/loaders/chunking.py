"""청킹 전략·청크 후처리 전용 믹스인"""

from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings
import logging

logger = logging.getLogger(__name__)


class ChunkingMixin:
    """청킹 전략·청크 후처리 전용 믹스인 — EnhancedDocumentLoader 믹스인"""

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
