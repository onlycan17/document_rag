"""
토큰 카운터 유틸리티
각 LLM 프로바이더별로 토큰을 계산하고 컨텍스트 윈도우 사용량을 추적합니다.
"""
import tiktoken
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TokenCounter:
    def __init__(self):
        """토큰 카운터 초기화"""
        # 각 프로바이더별 모델의 최대 컨텍스트 크기 (토큰)
        self.max_context_sizes = {
            "openai": {
                "gpt-3.5-turbo": 16385,
                "gpt-4": 8192,
                "gpt-4-turbo-preview": 128000,
                "gpt-4o": 128000,
                "gpt-4o-mini": 128000,
                "gpt-4.1": 1000000,
                "gpt-4.1-mini": 1000000,
                "gpt-4.1-nano": 1000000
            },
            "google": {
                "gemini-1.5-flash": 1048576,  # 1M tokens
                "gemini-1.5-flash-8b": 1048576,  # 1M tokens
                "gemini-2.5-flash": 1048576,  # 1M tokens
                "gemini-1.5-pro": 2097152,  # 2M tokens
                "gemini-2.5-pro": 2097152,  # 2M tokens
                "gemini-1.0-pro": 32760
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": 200000,
                "claude-3-haiku-20240307": 200000,
                "claude-3-sonnet-20240229": 200000,
                "claude-3-opus-20240229": 200000
            },
            "local": {
                "local-model": 4096  # 기본값, 실제 모델에 따라 다름
            }
        }
        
        # tiktoken 인코더 캐시
        self._encoders = {}
        
    def get_encoder(self, model_name: str):
        """모델에 맞는 tiktoken 인코더 반환"""
        if model_name not in self._encoders:
            try:
                # GPT 모델용 인코더
                if "gpt-4" in model_name:
                    self._encoders[model_name] = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in model_name:
                    self._encoders[model_name] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # 기본 인코더 (cl100k_base)
                    self._encoders[model_name] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to get encoder for {model_name}: {e}")
                self._encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoders[model_name]
    
    def count_tokens(self, text: str, model_name: str = "gpt-3.5-turbo") -> int:
        """텍스트의 토큰 수 계산"""
        try:
            encoder = self.get_encoder(model_name)
            return len(encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # 대략적인 추정 (4 글자당 1 토큰)
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model_name: str = "gpt-3.5-turbo") -> int:
        """메시지 리스트의 총 토큰 수 계산"""
        total_tokens = 0
        
        # 메시지별 오버헤드 (OpenAI 기준)
        if "gpt-3.5" in model_name or "gpt-4" in model_name:
            tokens_per_message = 3  # <|start|>{role}\n{content}<|end|>\n
            tokens_per_name = 1
        else:
            tokens_per_message = 4
            tokens_per_name = -1
        
        for message in messages:
            total_tokens += tokens_per_message
            
            # 역할과 내용 토큰 계산
            if "role" in message:
                total_tokens += self.count_tokens(message["role"], model_name)
            if "content" in message:
                total_tokens += self.count_tokens(message["content"], model_name)
            if "name" in message:
                total_tokens += tokens_per_name
                total_tokens += self.count_tokens(message["name"], model_name)
        
        # 응답을 위한 토큰 예약
        total_tokens += 3  # assistant 메시지 시작
        
        return total_tokens
    
    def get_max_context_size(self, provider: str, model: str) -> int:
        """프로바이더와 모델에 따른 최대 컨텍스트 크기 반환"""
        if provider in self.max_context_sizes:
            if model in self.max_context_sizes[provider]:
                return self.max_context_sizes[provider][model]
            else:
                # 해당 프로바이더의 기본값 사용
                return list(self.max_context_sizes[provider].values())[0]
        
        # 알 수 없는 프로바이더의 경우 기본값
        return 4096
    
    def calculate_context_usage(self, messages: List[Dict[str, str]], 
                              documents_text: str,
                              provider: str, 
                              model: str) -> Dict[str, any]:
        """컨텍스트 사용량 계산"""
        # 모델명 결정 (토큰 계산용)
        token_model = model if provider == "openai" else "gpt-3.5-turbo"
        
        # 메시지 토큰 계산
        messages_tokens = self.count_messages_tokens(messages, token_model)
        
        # 문서 토큰 계산
        documents_tokens = self.count_tokens(documents_text, token_model)
        
        # 시스템 프롬프트 토큰 계산 (대략적)
        system_prompt_tokens = 500  # 예상값
        
        # 총 사용 토큰
        total_tokens = messages_tokens + documents_tokens + system_prompt_tokens
        
        # 최대 컨텍스트 크기
        max_tokens = self.get_max_context_size(provider, model)
        
        # 사용률 계산
        usage_percentage = (total_tokens / max_tokens) * 100 if max_tokens > 0 else 100
        
        # 남은 토큰 (응답을 위한 공간)
        remaining_tokens = max_tokens - total_tokens
        
        return {
            "messages_tokens": messages_tokens,
            "documents_tokens": documents_tokens,
            "system_prompt_tokens": system_prompt_tokens,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "remaining_tokens": remaining_tokens,
            "usage_percentage": usage_percentage,
            "is_over_limit": total_tokens > max_tokens
        }
    
    def format_token_display(self, usage_info: Dict[str, any]) -> str:
        """토큰 사용량을 보기 좋은 형식으로 변환"""
        if usage_info["is_over_limit"]:
            status = "🔴 초과"
        elif usage_info["usage_percentage"] > 80:
            status = "🟡 주의"
        else:
            status = "🟢 정상"
        
        return f"{status} {usage_info['total_tokens']:,} / {usage_info['max_tokens']:,} 토큰 ({usage_info['usage_percentage']:.1f}%)"
    
    def get_usage_breakdown(self, usage_info: Dict[str, any]) -> str:
        """토큰 사용량 상세 분석"""
        breakdown = f"""
**토큰 사용량 분석**
- 대화 내역: {usage_info['messages_tokens']:,} 토큰
- 검색된 문서: {usage_info['documents_tokens']:,} 토큰
- 시스템 프롬프트: {usage_info['system_prompt_tokens']:,} 토큰
- **총 사용량**: {usage_info['total_tokens']:,} 토큰
- **남은 토큰**: {usage_info['remaining_tokens']:,} 토큰
"""
        return breakdown