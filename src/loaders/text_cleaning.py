"""텍스트 정리·한국어 정규화 전용 믹스인"""

import re
import logging

logger = logging.getLogger(__name__)


class TextCleaningMixin:
    """텍스트 정리·한국어 정규화 전용 믹스인 — EnhancedDocumentLoader 믹스인"""

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
