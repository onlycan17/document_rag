"""
태그 처리 모듈
스트리밍 중 특수 태그들(<think>, <system-reminder> 등)을 파싱하고 처리
"""

import re
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TagType(Enum):
    """태그 유형"""

    THINKING = "thinking"
    SYSTEM_REMINDER = "system_reminder"
    FUNCTION_CALLS = "function_calls"
    FUNCTION_RESULTS = "function_results"
    UNKNOWN = "unknown"


@dataclass
class TagContent:
    """태그 내용 정보"""

    tag_type: TagType
    content: str
    start_pos: int
    end_pos: int
    tag_name: str


class StreamingTagProcessor:
    """
    스트리밍 중 태그를 실시간으로 처리하는 클래스
    """

    def __init__(self, show_thinking: bool = False, show_system: bool = False):
        """
        Args:
            show_thinking: thinking 내용 표시 여부
            show_system: system-reminder 내용 표시 여부
        """
        self.show_thinking = show_thinking
        self.show_system = show_system

        # 누적 텍스트 버퍼
        self.buffer = ""
        self.processed_text = ""
        self.thinking_content = []
        self.system_content = []

        # 현재 태그 상태
        self.in_thinking = False
        self.in_system = False
        self.current_thinking = ""
        self.current_system = ""

        logger.debug(f"TagProcessor 초기화: thinking={show_thinking}, system={show_system}")

    def process_chunk(self, chunk: str) -> Dict[str, Any]:
        """
        스트리밍 청크를 처리하여 태그를 분리

        Args:
            chunk: 새로운 텍스트 청크

        Returns:
            Dict: 처리 결과
            {
                "display_text": str,  # 사용자에게 표시할 텍스트
                "thinking_text": str,  # thinking 내용 (있는 경우)
                "system_text": str,   # system-reminder 내용 (있는 경우)
                "has_thinking": bool,  # thinking 태그 포함 여부
                "has_system": bool     # system-reminder 태그 포함 여부
            }
        """
        self.buffer += chunk

        # 처리 결과 초기화
        result = {
            "display_text": "",
            "thinking_text": "",
            "system_text": "",
            "has_thinking": False,
            "has_system": False,
            "raw_chunk": chunk,
        }

        # 완료된 태그들을 찾아서 처리
        self._process_complete_tags()

        # 현재 진행 중인 태그 상태 확인
        current_display = self._filter_current_content()

        result["display_text"] = current_display
        result["thinking_text"] = self.current_thinking
        result["system_text"] = self.current_system
        result["has_thinking"] = self.in_thinking or bool(self.thinking_content)
        result["has_system"] = self.in_system or bool(self.system_content)

        return result

    def _process_complete_tags(self) -> str:
        """완료된 태그들을 처리"""
        processed = self.buffer

        # <think>...</think> 태그 처리
        think_pattern = r"<think>(.*?)</think>"
        for match in re.finditer(think_pattern, processed, re.DOTALL):
            thinking_content = match.group(1).strip()
            self.thinking_content.append(thinking_content)
            logger.debug(f"완료된 thinking 태그 발견: {len(thinking_content)}자")

            # 태그 제거
            processed = processed.replace(match.group(0), "")

        # <system-reminder>...</system-reminder> 태그 처리
        system_pattern = r"<system-reminder>(.*?)</system-reminder>"
        for match in re.finditer(system_pattern, processed, re.DOTALL):
            system_content = match.group(1).strip()
            self.system_content.append(system_content)
            logger.debug(f"완료된 system-reminder 태그 발견: {len(system_content)}자")

            # 태그 제거
            processed = processed.replace(match.group(0), "")

        self.buffer = processed
        return processed

    def _filter_current_content(self) -> str:
        """현재 버퍼에서 진행 중인 태그 상태를 확인하고 표시할 텍스트 반환"""
        content = self.buffer

        # <think> 태그 시작 확인
        if "<think>" in content:
            think_start = content.rfind("<think>")
            if "</think>" not in content[think_start:]:
                # thinking 태그가 열려있는 상태
                self.in_thinking = True
                self.current_thinking = content[think_start + 7 :]  # '<think>' 길이만큼 제외

                # thinking 태그 이전 내용만 표시
                return content[:think_start]

        # </think> 태그 확인
        if "</think>" in content and self.in_thinking:
            think_end = content.rfind("</think>")
            self.in_thinking = False
            # thinking 내용을 완료된 목록에 추가
            if self.current_thinking:
                full_thinking = self.current_thinking + content[:think_end]
                self.thinking_content.append(full_thinking.strip())
                self.current_thinking = ""

            # </think> 이후 내용 반환
            return content[think_end + 8 :]  # '</think>' 길이만큼 제외

        # <system-reminder> 태그 처리 (동일한 로직)
        if "<system-reminder>" in content:
            system_start = content.rfind("<system-reminder>")
            if "</system-reminder>" not in content[system_start:]:
                self.in_system = True
                self.current_system = content[system_start + 18 :]  # '<system-reminder>' 길이
                return content[:system_start]

        if "</system-reminder>" in content and self.in_system:
            system_end = content.rfind("</system-reminder>")
            self.in_system = False
            if self.current_system:
                full_system = self.current_system + content[:system_end]
                self.system_content.append(full_system.strip())
                self.current_system = ""
            return content[system_end + 19 :]  # '</system-reminder>' 길이

        # 태그 안에 있는 경우 빈 문자열 반환
        if self.in_thinking or self.in_system:
            return ""

        # 일반 텍스트 반환
        return content

    def get_thinking_summary(self) -> str:
        """thinking 내용 요약 반환"""
        if not self.thinking_content:
            return ""

        total_length = sum(len(content) for content in self.thinking_content)
        return f"💭 추론 과정: {len(self.thinking_content)}개 블록, 총 {total_length}자"

    def get_full_thinking(self) -> str:
        """전체 thinking 내용 반환"""
        if not self.thinking_content:
            return ""

        return "\n\n".join(f"💭 **추론 {i+1}**\n{content}" for i, content in enumerate(self.thinking_content))

    def get_system_messages(self) -> List[str]:
        """시스템 메시지 목록 반환"""
        return self.system_content.copy()

    def reset(self):
        """상태 초기화"""
        self.buffer = ""
        self.processed_text = ""
        self.thinking_content = []
        self.system_content = []
        self.in_thinking = False
        self.in_system = False
        self.current_thinking = ""
        self.current_system = ""
        logger.debug("TagProcessor 상태 초기화")

    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            "total_thinking_blocks": len(self.thinking_content),
            "total_system_messages": len(self.system_content),
            "current_thinking_length": len(self.current_thinking),
            "current_system_length": len(self.current_system),
            "buffer_length": len(self.buffer),
            "in_thinking": self.in_thinking,
            "in_system": self.in_system,
        }
