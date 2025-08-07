"""
한국어 텍스트 모델 - Midm-2.0-Base-Instruct GGUF 모델 사용

이 모듈은 KT의 한국어-영어 바이링귀 모델을 사용하여
문서의 주제와 핵심 내용을 추출합니다.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
from llama_cpp import Llama

logger = logging.getLogger(__name__)


class KoreanTextModel:
    """Midm-2.0을 사용한 한국어 텍스트 분석"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 n_ctx: int = 4096,
                 n_threads: int = 4,
                 n_gpu_layers: int = 0):
        """
        한국어 텍스트 모델 초기화
        
        Args:
            model_path: GGUF 모델 파일 경로
            n_ctx: 컨텍스트 크기 (토큰 수)
            n_threads: CPU 스레드 수
            n_gpu_layers: GPU로 옮길 레이어 수 (0=CPU only)
        """
        self.model_path = self._get_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        
        self.model = None
        self._load_model()
        
    def _get_model_path(self, model_path: Optional[str] = None) -> Path:
        """모델 경로 가져오기"""
        if model_path:
            return Path(model_path)
        
        # 기본 경로
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / "models" / "korean" / "Midm-2.0-Base-Instruct-Q4_K_S.gguf"
        
        if not default_path.exists():
            logger.warning(f"⚠️ 모델 파일이 없습니다: {default_path}")
            logger.info("scripts/download_models.py를 실행하여 모델을 다운로드하세요.")
            
        return default_path
    
    def _load_model(self):
        """모델 로드"""
        try:
            if not self.model_path.exists():
                logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {self.model_path}")
                return
            
            logger.info(f"🔄 한국어 모델 로딩 중: {self.model_path.name}")
            logger.info(f"   컨텍스트 크기: {self.n_ctx} 토큰")
            logger.info(f"   CPU 스레드: {self.n_threads}")
            
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            
            logger.info("✅ 한국어 모델 로드 완료!")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            self.model = None
    
    def extract_document_topic(self, text: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        문서의 주제와 핵심 내용 추출
        
        Args:
            text: 분석할 텍스트
            max_length: 최대 텍스트 길이
            
        Returns:
            주제와 핵심 내용 딕셔너리
        """
        if not self.model:
            logger.error("모델이 로드되지 않았습니다.")
            return {"error": "Model not loaded"}
        
        # 텍스트 길이 제한
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # 프롬프트 구성
        prompt = self._create_topic_extraction_prompt(text)
        
        try:
            # 모델 추론
            response = self.model(
                prompt,
                max_tokens=512,
                temperature=0.3,
                top_p=0.9,
                stop=["</answer>", "\n\n\n"]
            )
            
            # 결과 파싱
            result_text = response['choices'][0]['text'].strip()
            
            # JSON 형식으로 파싱 시도
            try:
                # JSON 블록 추출
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    json_text = result_text[json_start:json_end].strip()
                    return json.loads(json_text)
                else:
                    # 일반 텍스트를 구조화
                    return self._parse_text_response(result_text)
                    
            except json.JSONDecodeError:
                return self._parse_text_response(result_text)
                
        except Exception as e:
            logger.error(f"주제 추출 실패: {e}")
            return {"error": str(e)}
    
    def _create_topic_extraction_prompt(self, text: str) -> str:
        """주제 추출 프롬프트 생성"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 한국어 문서 분석 전문가입니다. 주어진 텍스트에서 주제와 핵심 내용을 추출하여 구조화된 JSON 형식으로 응답합니다.

다음 형식으로 응답하세요:
```json
{{
    "main_topic": "문서의 핵심 주제",
    "sub_topics": ["세부 주제1", "세부 주제2"],
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "summary": "50자 이내의 간단한 요약",
    "domain": "문서 분야 (예: 역사, 과학, 기술, 법률 등)"
}}
```
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
다음 텍스트를 분석해주세요:

{text[:1500]}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
텍스트를 분석하여 주제와 핵심 내용을 추출하겠습니다.

"""
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """텍스트 응답을 구조화된 형식으로 파싱"""
        result = {
            "main_topic": "",
            "sub_topics": [],
            "keywords": [],
            "summary": "",
            "domain": "",
            "raw_response": text
        }
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if '주제' in line and ':' in line:
                result["main_topic"] = line.split(':', 1)[1].strip()
            elif '키워드' in line and ':' in line:
                keywords_text = line.split(':', 1)[1].strip()
                result["keywords"] = [k.strip() for k in keywords_text.split(',')]
            elif '요약' in line and ':' in line:
                result["summary"] = line.split(':', 1)[1].strip()
            elif '분야' in line and ':' in line:
                result["domain"] = line.split(':', 1)[1].strip()
        
        # 기본값 설정
        if not result["main_topic"] and text:
            result["main_topic"] = text.split('.')[0].strip()
        
        return result
    
    def analyze_image_relevance(self, document_topic: Dict[str, Any], image_context: str) -> float:
        """
        이미지가 문서 주제와 얼마나 관련있는지 점수 계산
        
        Args:
            document_topic: 문서 주제 정보
            image_context: 이미지 주변 텍스트 컨텍스트
            
        Returns:
            관련도 점수 (0.0 ~ 1.0)
        """
        if not self.model:
            return 0.5  # 기본값
        
        # 프롬프트 구성
        prompt = self._create_relevance_prompt(document_topic, image_context)
        
        try:
            response = self.model(
                prompt,
                max_tokens=50,
                temperature=0.1,
                top_p=0.9
            )
            
            result_text = response['choices'][0]['text'].strip()
            
            # 점수 추출
            for line in result_text.split('\n'):
                if '점수' in line or 'score' in line.lower():
                    # 숫자 추출
                    import re
                    numbers = re.findall(r'[\d.]+', line)
                    if numbers:
                        score = float(numbers[0])
                        # 0-1 범위로 정규화
                        if score > 1:
                            score = score / 100
                        return min(max(score, 0.0), 1.0)
            
            # 키워드 기반 점수 계산 (폴백)
            return self._calculate_keyword_relevance(document_topic, image_context)
            
        except Exception as e:
            logger.error(f"관련도 분석 실패: {e}")
            return 0.5
    
    def _create_relevance_prompt(self, document_topic: Dict[str, Any], image_context: str) -> str:
        """관련도 평가 프롬프트 생성"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
이미지와 문서의 관련도를 0에서 1 사이의 점수로 평가합니다.
1에 가까울수록 매우 관련이 있고, 0에 가까울수록 관련이 없습니다.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
문서 주제: {document_topic.get('main_topic', '')}
키워드: {', '.join(document_topic.get('keywords', [])[:5])}

이미지 주변 텍스트:
{image_context[:300]}

이미지가 문서 주제와 얼마나 관련있는지 0-1 사이 점수로 평가해주세요.
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
점수: """
    
    def _calculate_keyword_relevance(self, document_topic: Dict[str, Any], image_context: str) -> float:
        """키워드 기반 관련도 계산 (폴백)"""
        if not image_context:
            return 0.3
        
        context_lower = image_context.lower()
        score = 0.0
        
        # 주제 매칭
        main_topic = document_topic.get('main_topic', '').lower()
        if main_topic and main_topic in context_lower:
            score += 0.4
        
        # 키워드 매칭
        keywords = document_topic.get('keywords', [])
        if keywords:
            matched_keywords = sum(1 for kw in keywords if kw.lower() in context_lower)
            score += min(0.4, matched_keywords * 0.1)
        
        # 세부 주제 매칭
        sub_topics = document_topic.get('sub_topics', [])
        if sub_topics:
            matched_topics = sum(1 for topic in sub_topics if topic.lower() in context_lower)
            score += min(0.2, matched_topics * 0.1)
        
        return min(score, 1.0)
    
    def generate_query_expansion(self, query: str) -> List[str]:
        """
        쿼리 확장 - 관련 키워드 생성
        
        Args:
            query: 원본 쿼리
            
        Returns:
            확장된 키워드 리스트
        """
        if not self.model:
            return [query]
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
주어진 검색어와 관련된 동의어, 유사어, 관련 용어를 5개 생성합니다.
한국어로만 응답하고, 쉼표로 구분합니다.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
검색어: {query}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
관련 용어: """
        
        try:
            response = self.model(
                prompt,
                max_tokens=50,
                temperature=0.5,
                top_p=0.9
            )
            
            result = response['choices'][0]['text'].strip()
            
            # 쉼표로 분리
            expanded_terms = [term.strip() for term in result.split(',')]
            
            # 원본 쿼리 포함
            if query not in expanded_terms:
                expanded_terms.insert(0, query)
            
            return expanded_terms[:6]  # 최대 6개
            
        except Exception as e:
            logger.error(f"쿼리 확장 실패: {e}")
            return [query]
    
    def cleanup(self):
        """모델 정리"""
        if self.model:
            del self.model
            self.model = None
            logger.info("한국어 모델 메모리 해제 완료")


# 편의 함수
def create_korean_text_model(model_path: Optional[str] = None) -> KoreanTextModel:
    """한국어 텍스트 모델 생성"""
    return KoreanTextModel(model_path=model_path)


if __name__ == "__main__":
    # 테스트
    model = create_korean_text_model()
    
    # 테스트 텍스트
    test_text = """
    몽촌토성은 서울특별시 송파구에 위치한 백제 초기의 토성으로,
    한성백제 시대의 중요한 방어시설이었습니다. 
    이 토성은 백제가 한강 유역을 지배하던 시기의 역사를 보여주는 중요한 유적입니다.
    """
    
    # 주제 추출 테스트
    topic = model.extract_document_topic(test_text)
    print("주제 추출 결과:")
    print(json.dumps(topic, ensure_ascii=False, indent=2))
    
    # 쿼리 확장 테스트
    expanded = model.generate_query_expansion("몽촌토성")
    print("\n쿼리 확장 결과:")
    print(expanded)
    
    # 정리
    model.cleanup()