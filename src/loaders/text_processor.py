"""
텍스트 처리 모듈

이 모듈은 기존 document_loader.py에서 텍스트 정제, 정규화, 마크다운 처리, 한국어 최적화 로직을 분리하여
단일 책임 원칙에 따라 독립적으로 관리합니다.

주요 기능:
- 텍스트 정제 및 정규화
- 마크다운 특화 후처리
- 한국어 텍스트 최적화
- 구조 정보 보존
- 중복 제거 및 최적화
"""

import re
import logging
from typing import List, Dict, Any
from langchain.schema import Document

from .text_patterns import (
    CLEANING_PATTERNS,
    KOREAN_PATTERNS,
    MARKDOWN_PATTERNS,
    STRUCTURE_PATTERNS,
)

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    텍스트 처리 및 정제 클래스

    기존 document_loader.py에서 텍스트 처리 로직을 분리하여
    재사용 가능하고 테스트 가능한 모듈로 구성
    """

    def __init__(self):
        """TextProcessor 초기화"""
        # 마크다운 패턴들
        self.markdown_patterns = MARKDOWN_PATTERNS

        # 정제 패턴들
        self.cleaning_patterns = CLEANING_PATTERNS

        # 한국어 처리 패턴들
        self.korean_patterns = KOREAN_PATTERNS

        # 구조 보존 패턴들
        self.structure_patterns = STRUCTURE_PATTERNS

        logger.info("TextProcessor 초기화 완료")

    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        텍스트 기본 정제

        Args:
            text: 정제할 텍스트
            preserve_structure: 구조 정보 보존 여부

        Returns:
            정제된 텍스트
        """
        if not text or not text.strip():
            return ""

        original_text = text

        try:
            # HTML 태그 및 엔티티 제거
            text = self.cleaning_patterns["html_tags"].sub("", text)
            text = self.cleaning_patterns["html_entities"].sub("", text)

            # 불필요한 공백 정리 (구조 보존 모드에서는 신중하게)
            if preserve_structure:
                # 구조를 해치지 않는 범위에서만 정리
                text = self.cleaning_patterns["trailing_spaces"].sub("", text)
                text = self.cleaning_patterns["multiple_newlines"].sub("\n\n", text)
            else:
                # 적극적인 공백 정리
                text = self.cleaning_patterns["multiple_spaces"].sub(" ", text)
                text = self.cleaning_patterns["multiple_newlines"].sub("\n\n", text)
                text = self.cleaning_patterns["trailing_spaces"].sub("", text)

            # 페이지 번호 및 각주 제거
            text = self.cleaning_patterns["page_numbers"].sub("", text)
            text = self.cleaning_patterns["footnotes"].sub("", text)

            # 중복 문장부호 정리
            text = self.cleaning_patterns["multiple_punctuation"].sub(r"\1", text)

            # 최종 정리
            text = text.strip()

            if not text:
                logger.warning("텍스트 정제 후 내용이 비어있음")
                return original_text.strip()

            return text

        except Exception as e:
            logger.error(f"텍스트 정제 중 오류: {str(e)}")
            return original_text.strip()

    def normalize_korean_text(self, text: str) -> str:
        """
        한국어 텍스트 정규화

        Args:
            text: 정규화할 텍스트

        Returns:
            정규화된 텍스트
        """
        if not text:
            return ""

        try:
            # 한국어 문장부호 정규화
            for old_punct, new_punct in self.korean_patterns["korean_punctuation"].items():
                text = text.replace(old_punct, new_punct)

            # 한국어 숫자 처리 (필요시)
            # 현재는 보존하지만, 향후 아라비아 숫자로 변환 로직 추가 가능

            # 한국어 단어 경계 정리
            text = re.sub(r"([가-힣])([a-zA-Z])", r"\1 \2", text)
            text = re.sub(r"([a-zA-Z])([가-힣])", r"\1 \2", text)

            # 한국어 문장에서 불필요한 공백 제거
            text = re.sub(r"([가-힣])\s+([.,!?;:])", r"\1\2", text)

            return text

        except Exception as e:
            logger.error(f"한국어 텍스트 정규화 중 오류: {str(e)}")
            return text

    def process_markdown_content(self, text: str) -> Dict[str, Any]:
        """
        마크다운 콘텐츠 처리 및 구조 추출

        Args:
            text: 마크다운 텍스트

        Returns:
            처리된 텍스트와 구조 정보
        """
        if not text:
            return {"text": "", "structure": {}}

        try:
            structure = {"headers": [], "code_blocks": [], "links": [], "images": [], "lists": [], "tables": []}

            # 헤더 추출
            for match in self.markdown_patterns["headers"].finditer(text):
                level = len(match.group(1))
                title = match.group(2).strip()
                structure["headers"].append({"level": level, "title": title, "position": match.start()})

            # 코드 블록 추출 (보존)
            code_blocks = []
            for match in self.markdown_patterns["code_blocks"].finditer(text):
                code_blocks.append({"content": match.group(0), "position": match.span()})
                structure["code_blocks"].append(match.group(0))

            # 링크 추출
            for match in self.markdown_patterns["links"].finditer(text):
                structure["links"].append({"text": match.group(1), "url": match.group(2)})

            # 이미지 추출
            for match in self.markdown_patterns["images"].finditer(text):
                structure["images"].append({"alt": match.group(1), "src": match.group(2)})

            # 리스트 항목 추출
            for match in self.markdown_patterns["unordered_lists"].finditer(text):
                structure["lists"].append({"type": "unordered", "content": match.group(1)})

            for match in self.markdown_patterns["ordered_lists"].finditer(text):
                structure["lists"].append({"type": "ordered", "content": match.group(1)})

            # 테이블 추출
            table_lines = []
            for match in self.markdown_patterns["tables"].finditer(text):
                table_lines.append(match.group(0))

            if table_lines:
                structure["tables"] = table_lines

            # 마크다운 구문을 일반 텍스트로 변환
            processed_text = self._convert_markdown_to_text(text, code_blocks)

            return {"text": processed_text, "structure": structure}

        except Exception as e:
            logger.error(f"마크다운 처리 중 오류: {str(e)}")
            return {"text": text, "structure": {}}

    def _convert_markdown_to_text(self, text: str, code_blocks: List[Dict]) -> str:
        """마크다운 구문을 일반 텍스트로 변환"""
        try:
            # 코드 블록은 보존
            protected_blocks = {}
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                protected_blocks[placeholder] = block["content"]
                text = text.replace(block["content"], placeholder)

            # 헤더 변환
            text = self.markdown_patterns["headers"].sub(r"\2", text)

            # 링크 변환 (텍스트만 남김)
            text = self.markdown_patterns["links"].sub(r"\1", text)

            # 이미지 변환 (alt 텍스트만 남김)
            text = self.markdown_patterns["images"].sub(r"\1", text)

            # 강조 제거
            text = self.markdown_patterns["bold"].sub(r"\1", text)
            text = self.markdown_patterns["italic"].sub(r"\1", text)

            # 인라인 코드 변환
            text = self.markdown_patterns["inline_code"].sub(r"\1", text)

            # 리스트 마커 제거
            text = self.markdown_patterns["unordered_lists"].sub(r"\1", text)
            text = self.markdown_patterns["ordered_lists"].sub(r"\1", text)

            # 인용구 마커 제거
            text = self.markdown_patterns["blockquotes"].sub(r"\1", text)

            # 구분선 제거
            text = self.markdown_patterns["horizontal_rules"].sub("", text)

            # 코드 블록 복원
            for placeholder, content in protected_blocks.items():
                text = text.replace(placeholder, content)

            return text

        except Exception as e:
            logger.error(f"마크다운 변환 중 오류: {str(e)}")
            return text

    def extract_structure_info(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 구조 정보 추출

        Args:
            text: 분석할 텍스트

        Returns:
            구조 정보 딕셔너리
        """
        structure = {
            "sections": [],
            "subsections": [],
            "numbered_items": [],
            "bulleted_items": [],
            "tables": [],
            "citations": [],
            "references": [],
        }

        try:
            # 섹션 추출
            for match in self.structure_patterns["sections"].finditer(text):
                structure["sections"].append({"title": match.group(1).strip(), "position": match.start()})

            # 하위 섹션 추출
            for match in self.structure_patterns["subsections"].finditer(text):
                structure["subsections"].append({"title": match.group(1).strip(), "position": match.start()})

            # 번호 목록 항목 추출
            for match in self.structure_patterns["numbered_items"].finditer(text):
                structure["numbered_items"].append(
                    {"number": match.group(1), "content": match.group(2).strip(), "position": match.start()}
                )

            # 불릿 목록 항목 추출
            for match in self.structure_patterns["bulleted_items"].finditer(text):
                structure["bulleted_items"].append({"content": match.group(1).strip(), "position": match.start()})

            # 테이블 헤더 추출
            for match in self.structure_patterns["table_headers"].finditer(text):
                structure["tables"].append({"header": match.group(0).strip(), "position": match.start()})

            # 인용 추출
            for match in self.structure_patterns["citations"].finditer(text):
                structure["citations"].append({"citation": match.group(0).strip(), "position": match.start()})

            # 참조 섹션 추출
            for match in self.structure_patterns["references"].finditer(text):
                structure["references"].append({"reference": match.group(0).strip(), "position": match.start()})

            return structure

        except Exception as e:
            logger.error(f"구조 정보 추출 중 오류: {str(e)}")
            return structure

    def remove_duplicates(self, texts: List[str], similarity_threshold: float = 0.8) -> List[str]:
        """
        텍스트 리스트에서 중복 제거

        Args:
            texts: 텍스트 리스트
            similarity_threshold: 유사도 임계값

        Returns:
            중복이 제거된 텍스트 리스트
        """
        if not texts:
            return []

        try:
            unique_texts = []
            seen_texts = set()

            for text in texts:
                if not text or not text.strip():
                    continue

                # 간단한 정규화
                normalized = re.sub(r"\s+", " ", text.strip().lower())

                # 정확히 같은 텍스트는 제외
                if normalized in seen_texts:
                    continue

                # 유사한 텍스트 검사 (간단한 방식)
                is_duplicate = False
                for existing in unique_texts:
                    existing_normalized = re.sub(r"\s+", " ", existing.strip().lower())

                    # 길이 차이가 크면 스킵
                    if abs(len(normalized) - len(existing_normalized)) > max(
                        len(normalized), len(existing_normalized)
                    ) * (1 - similarity_threshold):
                        continue

                    # 간단한 유사도 계산 (Jaccard 유사도)
                    words1 = set(normalized.split())
                    words2 = set(existing_normalized.split())

                    if not words1 or not words2:
                        continue

                    jaccard_similarity = len(words1 & words2) / len(words1 | words2)

                    if jaccard_similarity >= similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_texts.append(text)
                    seen_texts.add(normalized)

            logger.info(f"중복 제거: {len(texts)} -> {len(unique_texts)} 텍스트")
            return unique_texts

        except Exception as e:
            logger.error(f"중복 제거 중 오류: {str(e)}")
            return texts

    def optimize_for_search(self, text: str) -> str:
        """
        검색 최적화를 위한 텍스트 처리

        Args:
            text: 최적화할 텍스트

        Returns:
            검색 최적화된 텍스트
        """
        if not text:
            return ""

        try:
            # 기본 정제
            optimized = self.clean_text(text, preserve_structure=False)

            # 한국어 정규화
            optimized = self.normalize_korean_text(optimized)

            # 검색 키워드 강화를 위한 처리
            # 숫자와 단위 사이 공백 정리
            optimized = re.sub(r"(\d+)\s*([가-힣]{1,2})", r"\1\2", optimized)

            # 연도 표기 정규화
            optimized = re.sub(r"(\d{4})\s*년", r"\1년", optimized)

            # 전문 용어 정리
            optimized = re.sub(r"([가-힣]+)\s*([가-힣]{1,2})", r"\1\2", optimized)

            return optimized.strip()

        except Exception as e:
            logger.error(f"검색 최적화 중 오류: {str(e)}")
            return text

    def process_documents(
        self,
        documents: List[Document],
        clean_text: bool = True,
        normalize_korean: bool = True,
        extract_structure: bool = False,
        optimize_search: bool = True,
        remove_duplicates: bool = True,
    ) -> List[Document]:
        """
        문서 리스트 일괄 처리

        Args:
            documents: 처리할 문서 리스트
            clean_text: 텍스트 정제 여부
            normalize_korean: 한국어 정규화 여부
            extract_structure: 구조 정보 추출 여부
            optimize_search: 검색 최적화 여부
            remove_duplicates: 중복 제거 여부

        Returns:
            처리된 문서 리스트
        """
        if not documents:
            return []

        try:
            processed_docs = []
            seen_contents = set()

            for doc in documents:
                if not doc.page_content or not doc.page_content.strip():
                    continue

                content = doc.page_content
                metadata = doc.metadata.copy() if doc.metadata else {}

                # 텍스트 정제
                if clean_text:
                    content = self.clean_text(content, preserve_structure=not optimize_search)

                # 한국어 정규화
                if normalize_korean:
                    content = self.normalize_korean_text(content)

                # 마크다운 처리
                if content.strip().startswith("#") or "```" in content:
                    md_result = self.process_markdown_content(content)
                    content = md_result["text"]
                    if extract_structure and md_result["structure"]:
                        metadata["markdown_structure"] = md_result["structure"]

                # 구조 정보 추출
                if extract_structure:
                    structure = self.extract_structure_info(content)
                    if any(structure.values()):
                        metadata["text_structure"] = structure

                # 검색 최적화
                if optimize_search:
                    content = self.optimize_for_search(content)

                # 중복 검사
                if remove_duplicates:
                    content_hash = hash(content.strip().lower())
                    if content_hash in seen_contents:
                        continue
                    seen_contents.add(content_hash)

                # 최종 검증
                if content and content.strip():
                    processed_docs.append(Document(page_content=content, metadata=metadata))

            logger.info(f"문서 처리 완료: {len(documents)} -> {len(processed_docs)} 문서")
            return processed_docs

        except Exception as e:
            logger.error(f"문서 일괄 처리 중 오류: {str(e)}")
            return documents

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        텍스트 통계 정보 생성

        Args:
            text: 분석할 텍스트

        Returns:
            통계 정보 딕셔너리
        """
        if not text:
            return {}

        try:
            stats = {
                "total_chars": len(text),
                "total_chars_no_spaces": len(text.replace(" ", "")),
                "total_words": len(text.split()),
                "total_lines": len(text.split("\n")),
                "total_sentences": len(re.findall(r"[.!?]+", text)),
                "korean_chars": len(re.findall(r"[가-힣]", text)),
                "english_chars": len(re.findall(r"[a-zA-Z]", text)),
                "numbers": len(re.findall(r"\d", text)),
                "punctuation": len(re.findall(r"[.,!?;:]", text)),
            }

            # 비율 계산
            if stats["total_chars"] > 0:
                stats["korean_ratio"] = stats["korean_chars"] / stats["total_chars"]
                stats["english_ratio"] = stats["english_chars"] / stats["total_chars"]
                stats["number_ratio"] = stats["numbers"] / stats["total_chars"]

            return stats

        except Exception as e:
            logger.error(f"텍스트 통계 생성 중 오류: {str(e)}")
            return {}
