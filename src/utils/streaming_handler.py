"""
스트리밍 핸들러 모듈
로컬 모델의 추론 내용을 깔끔하게 처리하기 위한 커스텀 핸들러
"""

from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from src.utils.logging_config import get_logger
from src.utils.tag_processor import StreamingTagProcessor

logger = get_logger(__name__)


class SilentStreamingHandler(BaseCallbackHandler):
    """
    조용한 스트리밍 핸들러
    터미널 출력 없이 스트리밍만 처리
    """
    
    def __init__(self, show_console_output: bool = False):
        """
        Args:
            show_console_output: 콘솔 출력 여부 (기본값: False)
        """
        self.show_console_output = show_console_output
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """새로운 토큰이 생성될 때 호출"""
        self.tokens.append(token)
        
        # 콘솔 출력이 활성화된 경우에만 출력
        if self.show_console_output:
            print(token, end="", flush=True)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 응답 완료 시 호출"""
        if self.show_console_output:
            print("\n")  # 줄바꿈
        
        # 토큰 리스트 초기화
        self.tokens = []
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 오류 발생 시 호출"""
        logger.error(f"LLM 스트리밍 오류: {str(error)}")
        self.tokens = []
    
    def get_accumulated_tokens(self) -> str:
        """축적된 토큰들을 문자열로 반환"""
        return ''.join(self.tokens)


class CustomStreamingHandler(BaseCallbackHandler):
    """
    커스텀 스트리밍 핸들러
    추론 내용을 선택적으로 표시하고 로깅 기능 제공
    """
    
    def __init__(self, show_reasoning: bool = False, show_tokens: bool = False):
        """
        Args:
            show_reasoning: 추론 과정 표시 여부
            show_tokens: 개별 토큰 표시 여부
        """
        self.show_reasoning = show_reasoning
        self.show_tokens = show_tokens
        self.current_text = ""
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 시작 시 호출"""
        if self.show_reasoning:
            logger.info("🤖 로컬 모델 추론 시작")
        
        self.current_text = ""
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """새로운 토큰이 생성될 때 호출"""
        self.current_text += token
        self.token_count += 1
        
        if self.show_tokens:
            # 토큰별 표시 (개발/디버그 용도)
            logger.debug(f"Token {self.token_count}: '{token}'")
        elif self.show_reasoning:
            # 추론 과정만 간단히 표시
            if self.token_count % 10 == 0:  # 10토큰마다
                logger.info(f"생성 중... ({self.token_count} 토큰)")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 응답 완료 시 호출"""
        if self.show_reasoning:
            logger.info(f"✅ 추론 완료 (총 {self.token_count} 토큰)")
            
            # 생성된 전체 텍스트의 요약 정보
            text_length = len(self.current_text)
            words_count = len(self.current_text.split())
            logger.info(f"📝 생성된 텍스트: {text_length}자, {words_count}단어")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 오류 발생 시 호출"""
        logger.error(f"❌ 로컬 모델 오류: {str(error)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """생성 통계 반환"""
        return {
            "token_count": self.token_count,
            "text_length": len(self.current_text),
            "word_count": len(self.current_text.split()) if self.current_text else 0,
            "current_text": self.current_text
        }


class TagAwareStreamingHandler(BaseCallbackHandler):
    """
    태그를 인식하는 스트리밍 핸들러
    <think>, <system-reminder> 등의 태그를 실시간으로 처리
    """
    
    def __init__(self, show_thinking: bool = False, show_console: bool = False, show_reasoning: bool = False):
        """
        Args:
            show_thinking: thinking 내용 표시 여부
            show_console: 콘솔 출력 여부
            show_reasoning: 추론 과정 표시 여부
        """
        self.show_thinking = show_thinking
        self.show_console = show_console
        self.show_reasoning = show_reasoning
        
        # 태그 프로세서 초기화
        self.tag_processor = StreamingTagProcessor(
            show_thinking=show_thinking,
            show_system=False  # system-reminder는 항상 숨김
        )
        
        # 통계 정보
        self.total_tokens = 0
        self.display_tokens = 0
        self.thinking_tokens = 0
        
        logger.info(f"TagAwareStreamingHandler 초기화: thinking={show_thinking}, console={show_console}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """LLM 시작 시 호출"""
        if self.show_reasoning:
            logger.info("🤖 태그 인식 스트리밍 시작")
        
        # 상태 초기화
        self.tag_processor.reset()
        self.total_tokens = 0
        self.display_tokens = 0
        self.thinking_tokens = 0
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """새로운 토큰이 생성될 때 호출"""
        self.total_tokens += 1
        
        # 태그 프로세서로 토큰 처리
        result = self.tag_processor.process_chunk(token)
        
        # 표시할 텍스트가 있는 경우
        if result["display_text"]:
            self.display_tokens += len(result["display_text"])
            
            # 콘솔 출력 (선택적)
            if self.show_console:
                print(result["display_text"], end="", flush=True)
        
        # thinking 내용 처리
        if result["has_thinking"] and self.show_thinking:
            thinking_text = result["thinking_text"]
            if thinking_text:
                self.thinking_tokens += len(thinking_text)
                
                if self.show_console:
                    print(f"\n💭 [추론 중: {len(thinking_text)}자]", end="", flush=True)
        
        # 디버그 정보 (선택적)
        if self.show_reasoning and self.total_tokens % 20 == 0:
            stats = self.tag_processor.get_stats()
            logger.debug(f"토큰 {self.total_tokens}: 표시={self.display_tokens}, 추론={self.thinking_tokens}, "
                        f"thinking_blocks={stats['total_thinking_blocks']}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """LLM 응답 완료 시 호출"""
        if self.show_console:
            print("\n")  # 줄바꿈
        
        stats = self.tag_processor.get_stats()
        
        if self.show_reasoning:
            logger.info(f"✅ 스트리밍 완료")
            logger.info(f"📊 토큰 통계: 전체={self.total_tokens}, 표시={self.display_tokens}, 추론={self.thinking_tokens}")
            logger.info(f"🧠 추론 통계: {stats['total_thinking_blocks']}개 블록")
            
            # thinking 요약 표시
            thinking_summary = self.tag_processor.get_thinking_summary()
            if thinking_summary:
                logger.info(f"💭 {thinking_summary}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """LLM 오류 발생 시 호출"""
        logger.error(f"❌ 태그 인식 스트리밍 오류: {str(error)}")
        
        if self.show_console:
            print(f"\n[오류: {str(error)}]\n")
    
    def get_thinking_content(self) -> str:
        """추론 내용 반환"""
        return self.tag_processor.get_full_thinking()
    
    def get_thinking_summary(self) -> str:
        """추론 요약 반환"""
        return self.tag_processor.get_thinking_summary()
    
    def get_stats(self) -> Dict[str, Any]:
        """상세 통계 반환"""
        tag_stats = self.tag_processor.get_stats()
        return {
            **tag_stats,
            "total_tokens": self.total_tokens,
            "display_tokens": self.display_tokens,
            "thinking_tokens": self.thinking_tokens,
            "thinking_ratio": self.thinking_tokens / max(self.total_tokens, 1) * 100
        }


def create_streaming_handler(mode: str = "silent", **kwargs) -> BaseCallbackHandler:
    """
    스트리밍 핸들러 팩토리 함수
    
    Args:
        mode: 핸들러 모드 ("silent", "reasoning", "debug", "tag_aware")
        **kwargs: 추가 설정
    
    Returns:
        BaseCallbackHandler: 설정된 스트리밍 핸들러
    """
    if mode == "tag_aware":
        return TagAwareStreamingHandler(
            show_thinking=kwargs.get("show_thinking", False),
            show_console=kwargs.get("show_console", False),
            show_reasoning=kwargs.get("show_reasoning", False)
        )
    
    elif mode == "silent":
        return SilentStreamingHandler(show_console_output=kwargs.get("show_console", False))
    
    elif mode == "reasoning":
        return CustomStreamingHandler(
            show_reasoning=True, 
            show_tokens=kwargs.get("show_tokens", False)
        )
    
    elif mode == "debug":
        return CustomStreamingHandler(
            show_reasoning=True, 
            show_tokens=True
        )
    
    else:
        # 기본값: 태그 인식 모드 (조용함)
        return TagAwareStreamingHandler(
            show_thinking=False,
            show_console=kwargs.get("show_console", False),
            show_reasoning=False
        )