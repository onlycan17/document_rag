"""
문서 구조 파싱 에이전트 - 제목, 목차, 표 등을 올바른 마크다운으로 변환
"""

import re
import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from config import settings

logger = logging.getLogger(__name__)


class StructureParserAgent(BaseAgent):
    """
    문서의 구조적 요소를 인식하고 올바른 마크다운 형식으로 변환하는 에이전트
    """

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        super().__init__("StructureParser", provider=provider, model_name=model_name)

        # 마크다운 구조 변환 예시들
        self.structure_examples = [
            "제목 인식: '1. 조사경위와 목적' → '# 1. 조사경위와 목적'",
            "소제목 인식: '가. 조사경위' → '## 가. 조사경위'",
            "잘못된 헤딩 수정: '### 다.' → '다.'",
            "목차 구조: '1) 개요' → '### 1) 개요'",
        ]

    def process(self, text_content: str) -> str:
        """
        텍스트 내용의 구조를 분석하여 올바른 마크다운으로 변환

        Args:
            text_content: 구조화할 텍스트 내용

        Returns:
            구조화된 마크다운 텍스트
        """
        # 1단계: 잘못된 헤딩 패턴 수정
        text_content = self._fix_incorrect_headings(text_content)

        # 2단계: 문서 구조 인식 및 헤딩 적용
        structured_content = self._apply_document_structure(text_content)

        # 3단계: 표 구조 정리
        structured_content = self._organize_tables(structured_content)

        return structured_content

    def _fix_incorrect_headings(self, content: str) -> str:
        """
        잘못 생성된 헤딩 패턴 수정
        """
        # "### 다." 같은 잘못된 헤딩 제거
        fixed_content = re.sub(r"^###\s*다\.\s*$", "다.", content, flags=re.MULTILINE)
        fixed_content = re.sub(r"^###\s*[가-힣]\.\s*$", lambda m: m.group(0)[4:], fixed_content, flags=re.MULTILINE)

        logger.info("🔧 잘못된 헤딩 패턴 수정 완료")
        return fixed_content

    def _apply_document_structure(self, content: str) -> str:
        """
        LLM을 활용하여 문서 구조 인식 및 헤딩 적용
        """
        # 텍스트를 단락별로 분할
        paragraphs = content.split("\n\n")
        structured_paragraphs = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # 구조적 요소인지 LLM으로 판단
            structure_type = self._identify_structure_type(paragraph)

            if structure_type["is_heading"]:
                # 헤딩 레벨에 따라 마크다운 적용
                level = structure_type["level"]
                heading_prefix = "#" * level + " "
                structured_paragraph = heading_prefix + paragraph.strip()
            else:
                structured_paragraph = paragraph

            structured_paragraphs.append(structured_paragraph)

        return "\n\n".join(structured_paragraphs)

    def _identify_structure_type(self, paragraph: str) -> Dict[str, Any]:
        """
        단락의 구조적 역할 식별
        """
        # 간단한 패턴부터 먼저 확인 (LLM 호출 최소화)
        first_line = paragraph.strip().split("\n")[0]

        # 명확한 제목 패턴들
        title_patterns = [
            r"^\d+\.\s+[가-힣]+",  # "1. 조사경위"
            r"^[가-힣]\.\s+[가-힣]+",  # "가. 조사지역"
            r"^\d+\)\s+[가-힣]+",  # "1) 개요"
            r"^[IVX]+\.\s+[가-힣]+",  # "I. 서론"
        ]

        for pattern in title_patterns:
            if re.match(pattern, first_line):
                # 숫자 깊이에 따라 레벨 결정
                if re.match(r"^\d+\.", first_line):
                    level = 1
                elif re.match(r"^[가-힣]\.", first_line):
                    level = 2
                else:
                    level = 3

                return {"is_heading": True, "level": level, "confidence": 0.9}

        # 패턴으로 판단 어려운 경우 LLM 사용
        return self._llm_identify_structure(paragraph)

    def _llm_identify_structure(self, paragraph: str) -> Dict[str, Any]:
        """
        LLM을 사용하여 구조 식별
        """
        first_line = paragraph.strip().split("\n")[0][:100]  # 첫 줄만 분석

        prompt = self._create_korean_prompt(
            "이 텍스트가 제목/소제목인지 본문인지 판단",
            f"텍스트: '{first_line}'",
            [
                "제목 예시: '1. 조사경위와 목적' → HEADING|1",
                "소제목 예시: '가. 조사지역' → HEADING|2",
                "본문 예시: '몽촌토성은 서울시 송파구...' → CONTENT|0",
            ],
        )

        prompt += """

다음 형식으로 응답하세요:
HEADING|레벨숫자 - 제목인 경우 (레벨: 1=대제목, 2=중제목, 3=소제목)
CONTENT|0 - 일반 본문인 경우

예시:
HEADING|1
CONTENT|0
"""

        try:
            response = self._call_llm(prompt, temperature=settings.temperature, max_tokens=50)

            # 첫 유효 라인만 사용 (LLM이 추가 텍스트를 반환해도 안전)
            first_resp_line = next((ln.strip() for ln in response.splitlines() if ln.strip()), "")

            # HEADING|N 패턴만 정규식으로 안전 파싱
            m = re.search(r"^HEADING\|(\d+)", first_resp_line)
            if m:
                try:
                    level = int(m.group(1))
                    # 레벨 범위 가드(1~6)
                    level = max(1, min(level, 6))
                    return {"is_heading": True, "level": level, "confidence": 0.8}
                except ValueError as err:
                    logger.debug(f"헤딩 레벨 파싱 실패(무시): {err}")

            # CONTENT|0 또는 기타 응답은 본문으로 처리
            return {"is_heading": False, "level": 0, "confidence": 0.8}

        except Exception as e:
            logger.warning(f"⚠️ 구조 식별 LLM 호출 실패: {str(e)}")
            return {"is_heading": False, "level": 0, "confidence": 0.3}

    def _organize_tables(self, content: str) -> str:
        """
        표 구조 정리 및 마크다운 테이블 형식 적용
        """
        # 테이블 패턴 감지 및 정리는 추후 구현
        # 현재는 기본 정리만 수행

        # 연속된 빈 줄 정리
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content
