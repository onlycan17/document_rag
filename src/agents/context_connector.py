"""
문맥 연결 에이전트 - 페이지 경계에서 끊어진 문장을 자연스럽게 연결
"""

import logging
from typing import List

from config import settings
from src.processing.chunked_cleanup import MAX_OUTPUT_TOKENS, clean_document
from src.utils.sentence_completion import join_page_boundaries

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ContextConnectorAgent(BaseAgent):
    """페이지별 텍스트를 규칙으로 잇고, 문장 경계 분할 LLM 교정으로 문맥을 연결하는 에이전트"""

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        super().__init__("ContextConnector", provider=provider, model_name=model_name)

    def process(self, text_blocks: List[str]) -> List[str]:
        """
        텍스트 블록들을 완전한 문장 단위로 재구성

        Args:
            text_blocks: 페이지별로 분리된 텍스트 블록들

        Returns:
            문장 단위로 재구성된 텍스트 블록들
        """
        if len(text_blocks) <= 1:
            return text_blocks

        full_text = "\n\n".join(page for page in join_page_boundaries(text_blocks) if page.strip())
        logger.info(f"📝 전체 텍스트 길이: {len(full_text)}자")

        result = clean_document(
            full_text,
            lambda prompt: self._call_llm(prompt, temperature=settings.temperature, max_tokens=MAX_OUTPUT_TOKENS),
            max_workers=settings.parallel_workers,
        )
        return self._split_into_blocks(result.text)

    def _split_into_blocks(self, text: str, max_block_size: int = 3000) -> List[str]:
        """재구성된 텍스트를 문단 경계에서 적절한 크기의 블록으로 분할"""
        if len(text) <= max_block_size:
            return [text]

        blocks = []
        current_block = ""
        for paragraph in text.split("\n\n"):
            if len(current_block) + len(paragraph) + 2 <= max_block_size:
                current_block = f"{current_block}\n\n{paragraph}" if current_block else paragraph
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = paragraph

        if current_block:
            blocks.append(current_block)

        logger.info(f"📋 텍스트 분할 완료: {len(blocks)}개 블록")
        return blocks
