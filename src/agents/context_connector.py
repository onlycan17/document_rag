"""
문맥 연결 에이전트 - 페이지 경계에서 끊어진 문장을 자연스럽게 연결
"""

import re
import logging
from typing import List
from .base_agent import BaseAgent
from config import settings

logger = logging.getLogger(__name__)


class ContextConnectorAgent(BaseAgent):
    """
    로컬 LLM을 활용하여 문맥이 끊어진 텍스트를 지능적으로 연결하는 에이전트
    """

    def __init__(self, provider: str | None = None, model_name: str | None = None):
        super().__init__("ContextConnector", provider=provider, model_name=model_name)

        # 한국어 문장 완성도 판단을 위한 예시들
        self.completion_examples = [
            "완전한 문장: '몽촌토성은 백제 한성시대의 왕성으로 추정된다.' → 완전함",
            "불완전한 문장: '몽촌토성은 백제 한성시대의' → 불완전함 (서술어 부족)",
            "불완전한 문장: '구분하고 있' → 불완전함 (어미 '다' 누락)",
            "불완전한 문장: '삼국사기 백' → 불완전함 ('제본기' 누락된 고유명사)",
        ]

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

        # 모든 텍스트를 하나로 합치기
        full_text = "\n".join(text_blocks)

        logger.info(f"📝 전체 텍스트 길이: {len(full_text)}자")

        # LLM으로 전체 텍스트를 문장 단위로 재구성
        restructured_text = self._restructure_full_text(full_text)

        # 재구성된 텍스트를 적절한 크기의 블록으로 분할
        return self._split_into_blocks(restructured_text)

    def _restructure_full_text(self, full_text: str) -> str:
        """
        전체 텍스트를 LLM으로 문장 단위로 재구성 (청크 단위 처리)
        """
        # 텍스트가 너무 크면 청크로 나누어 처리
        max_chunk_size = 3000  # 로컬 LLM 토큰 제한 고려

        if len(full_text) <= max_chunk_size:
            return self._process_text_chunk(full_text)

        # 청크로 분할하여 순차적으로 처리
        chunks = self._split_text_into_chunks(full_text, max_chunk_size)
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            logger.info(f"📝 청크 {i+1}/{len(chunks)} 처리 중...")
            processed_chunk = self._process_text_chunk(chunk)
            processed_chunks.append(processed_chunk)

        # 처리된 청크들을 다시 합치기
        result = "\n".join(processed_chunks)
        logger.info(f"✅ 전체 텍스트 재구성 완료: {len(result)}자")
        return result

    def _process_text_chunk(self, text_chunk: str) -> str:
        """
        텍스트 청크 하나를 LLM으로 처리
        """
        prompt = f"""PDF에서 추출된 텍스트의 문맥을 연결하고 띄어쓰기를 교정해주세요.

규칙:
1. '다.'로 끝나지 않는 불완전한 문장은 반드시 다음 텍스트와 연결
2. 띄어쓰기 오류 수정 (예: "운영에홍승연은" → "운영에 홍승연은")
3. 끊어진 단어/문장 연결 (예: "삼국사기 백제본기")
4. 제목, 목차는 그대로 유지
5. 원본 내용 변경 금지, 문맥 연결과 띄어쓰기만 수정

⚠️ 중요: 설명이나 부가적인 텍스트를 추가하지 마세요. 오직 교정된 텍스트만 출력하세요.

원본 텍스트:
{text_chunk}

교정된 텍스트(설명 없이 결과만):"""

        try:
            response = self._call_llm(prompt, temperature=settings.temperature, max_tokens=4000)

            if response.strip():
                # LLM 응답에서 불필요한 텍스트 제거
                cleaned = self._clean_llm_response(response.strip())
                return cleaned if cleaned else text_chunk
            else:
                logger.warning("⚠️ LLM 응답이 비어있음, 원본 반환")
                return text_chunk

        except Exception as e:
            logger.warning(f"⚠️ LLM 청크 처리 실패, 원본 반환: {str(e)}")
            return text_chunk

    def _clean_llm_response(self, text: str) -> str:
        """
        LLM 응답에서 불필요한 메타 텍스트 제거
        """

        # 제거할 패턴들 (더 포괄적으로)
        patterns_to_remove = [
            # LLM이 추가하는 설명 (다양한 변형 포함)
            r"한국어 문서 전문가로서[^\n]*\n?",
            r"한국문화사 문서에서[^\n]*제공합니다[^\n]*\n?",
            r"다음은[^\n]*텍스트입니다[^\n]*:\n?",
            r"교정된 텍스트[^\n]*:\n?",
            r"문맥을[^\n]*연결하겠습니다[^\n]*\n?",
            r"문맥을[^\n]*연결하여[^\n]*제공합니다[^\n]*\n?",
            r"텍스트의 문맥을[^\n]*연결[^\n]*\n?",
            # 섹션 헤더로 나타나는 메타 텍스트
            r"^#+\s*한국문화사 문서에서[^\n]*\n?",
            r"^#+\s*한국어 문서[^\n]*제공합니다[^\n]*\n?",
            # 구분선
            r"\n---+\s*한국[^\n]*\n---+\n?",
            r"\n---+\s*$",
        ]

        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

        # 연속된 줄바꿈 정리
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def _split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """
        텍스트를 지정된 크기의 청크로 분할 (문단 경계 고려)
        """
        if len(text) <= max_size:
            return [text]

        chunks = []
        lines = text.split("\n")
        current_chunk = ""

        for line in lines:
            # 현재 청크에 라인을 추가했을 때 크기 확인
            if len(current_chunk) + len(line) + 1 <= max_size:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
            else:
                # 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"📋 텍스트를 {len(chunks)}개 청크로 분할")
        return chunks

    def _split_into_blocks(self, text: str, max_block_size: int = 3000) -> List[str]:
        """
        재구성된 텍스트를 적절한 크기의 블록으로 분할
        """
        if len(text) <= max_block_size:
            return [text]

        blocks = []
        paragraphs = text.split("\n\n")  # 문단 단위로 분할
        current_block = ""

        for paragraph in paragraphs:
            if len(current_block) + len(paragraph) + 2 <= max_block_size:
                if current_block:
                    current_block += "\n\n" + paragraph
                else:
                    current_block = paragraph
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = paragraph

        if current_block:
            blocks.append(current_block)

        logger.info(f"📋 텍스트 분할 완료: {len(blocks)}개 블록")
        return blocks

    def _simple_connect(self, block1: str, block2: str) -> str:
        """
        간단한 블록 연결 (하위 호환성을 위해 유지)
        """
        return f"{block1.rstrip()}\n{block2.lstrip()}"
