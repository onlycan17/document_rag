"""
품질 검증 시스템 - MD 파일의 문맥 연결 품질을 분석하고 점수화
"""

import re
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

from src.agents.base_agent import LocalLLMAgent
from config import settings

logger = logging.getLogger(__name__)


class QualityChecker(LocalLLMAgent):
    """
    MD 파일 품질 검증 전문 클래스
    문맥 연결성, 띄어쓰기, 문장 완성도 등을 종합 평가
    """
    
    def __init__(self, provider: str | None = None, model_name: str | None = None, base_url: str | None = None):
        # 선택한 제공자/모델 수용(없으면 세션/설정)
        if provider is None:
            try:
                import streamlit as st  # type: ignore
                provider = st.session_state.get('current_provider', None)
                model_name = model_name or st.session_state.get('current_model', None)
            except Exception:
                pass
        super().__init__("QualityChecker", provider=provider, model_name=model_name, base_url=base_url)
        
        # 품질 평가 기준
        self.quality_criteria = {
            'sentence_completion': 0.4,  # 문장 완성도 (40%)
            'spacing_accuracy': 0.3,     # 띄어쓰기 정확도 (30%)
            'context_connection': 0.2,   # 문맥 연결성 (20%)
            'structure_preservation': 0.1 # 구조 보존도 (10%)
        }
        
        # 목표 점수
        self.target_score = 90.0
        
        logger.info("🔍 품질 검증 시스템 초기화 완료")
    
    def process(self, input_data: Any) -> Any:
        """
        에이전트 베이스 클래스의 추상 메서드 구현
        MD 파일 경로를 받아 품질 분석 결과 반환
        """
        if isinstance(input_data, str):
            return self.analyze_quality(input_data)
        else:
            raise ValueError("입력 데이터는 MD 파일 경로(문자열)여야 합니다")
    
    def analyze_quality(self, md_file_path: str) -> Dict[str, Any]:
        """
        MD 파일의 품질을 종합 분석
        
        Args:
            md_file_path: 분석할 MD 파일 경로
            
        Returns:
            품질 분석 결과 딕셔너리
        """
        md_path = Path(md_file_path)
        
        if not md_path.exists():
            raise FileNotFoundError(f"MD 파일을 찾을 수 없습니다: {md_file_path}")
        
        logger.info(f"🔍 품질 검증 시작: {md_path.name}")
        
        # MD 파일 읽기
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 각 항목별 점수 계산
        sentence_score = self._check_sentence_completion(content)
        spacing_score = self._check_spacing_accuracy(content)
        context_score = self._check_context_connection(content)
        structure_score = self._check_structure_preservation(content)
        
        # 가중 평균으로 총점 계산
        total_score = (
            sentence_score * self.quality_criteria['sentence_completion'] +
            spacing_score * self.quality_criteria['spacing_accuracy'] +
            context_score * self.quality_criteria['context_connection'] +
            structure_score * self.quality_criteria['structure_preservation']
        )
        
        # 개선 제안 생성
        improvements = self._generate_improvements(
            sentence_score, spacing_score, context_score, structure_score
        )
        
        result = {
            'total_score': round(total_score, 1),
            'scores': {
                'sentence_completion': round(sentence_score, 1),
                'spacing_accuracy': round(spacing_score, 1),
                'context_connection': round(context_score, 1),
                'structure_preservation': round(structure_score, 1)
            },
            'passed': total_score >= self.target_score,
            'target_score': self.target_score,
            'improvements': improvements,
            'file_stats': {
                'total_chars': len(content),
                'total_lines': len(content.split('\n')),
                'total_sentences': len(re.findall(r'[.!?]+', content))
            }
        }
        
        logger.info(f"📊 품질 점수: {total_score:.1f}/100 ({'통과' if result['passed'] else '재처리 필요'})")
        
        return result
    
    def _check_sentence_completion(self, content: str) -> float:
        """
        문장 완성도 검사 (불완전한 문장 비율 체크)
        """
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return 100.0
        
        incomplete_count = 0
        total_sentences = 0
        
        for line in lines:
            # 제목이나 목차는 제외
            if line.startswith('#') or re.match(r'^\d+\)', line) or '........' in line:
                continue
            
            total_sentences += 1
            
            # 불완전한 문장 패턴 감지
            if self._is_incomplete_sentence(line):
                incomplete_count += 1
        
        if total_sentences == 0:
            return 100.0
        
        completion_rate = (total_sentences - incomplete_count) / total_sentences * 100
        return max(0.0, completion_rate)
    
    def _is_incomplete_sentence(self, line: str) -> bool:
        """불완전한 문장인지 판단"""
        line = line.strip()
        
        if not line:
            return False
        
        # 완전한 문장 끝 패턴
        complete_endings = ['.', '!', '?', '다.', '음.', '였다.', '었다.', '한다.', '된다.', '이다.', '않다.', '습니다.', '니다.']
        for ending in complete_endings:
            if line.endswith(ending):
                return False
        
        # 불완전한 문장 패턴들
        incomplete_patterns = [
            r'.*하고\s*있$',      # "구분하고 있"
            r'.*되고\s*있$',      # "설치되고 있"
            r'.*하고$',           # "확인하고"
            r'.*되고$',           # "설치되고"
            r'.*하여$',           # "설치하여"
            r'.*되어$',           # "건설되어" 
            r'.*하는$',           # "만드는"
            r'.*되는$',           # "설치되는"
            r'.*을$',             # "만들"
            r'.*를$',             # "확인을"
            r'.*의$',             # "중요성의"
            r'.*는$',             # "오는"
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, line):
                return True
        
        # 조사로 끝나는 경우
        particles = ['를', '을', '의', '에', '서', '는', '이', '가', '와', '과', '로', '으로']
        for particle in particles:
            if line.endswith(particle):
                return True
        
        return False
    
    def _check_spacing_accuracy(self, content: str) -> float:
        """
        띄어쓰기 정확도 검사 (LLM 활용)
        """
        # 샘플 텍스트 추출 (전체를 다 보낼 수는 없으므로)
        sample_lines = []
        lines = content.split('\n')
        
        # 균등하게 샘플링 (최대 20줄)
        sample_interval = max(1, len(lines) // 20)
        for i in range(0, len(lines), sample_interval):
            line = lines[i].strip()
            if line and not line.startswith('#') and len(line) > 10:
                sample_lines.append(line)
                if len(sample_lines) >= 20:
                    break
        
        if not sample_lines:
            return 100.0
        
        sample_text = '\n'.join(sample_lines)
        
        prompt = f"""한국어 띄어쓰기 전문가로서, 다음 텍스트의 띄어쓰기 정확도를 0-100점으로 평가해주세요.

평가 기준:
- 명사와 조사 분리: "운영에 홍승연은" (정확) vs "운영에홍승연은" (오류)
- 복합명사: "삼국사기 백제본기" (정확) vs "삼국사기백제본기" (오류)  
- 동사 활용: "구분하고 있다" (정확) vs "구분하고있다" (오류)

텍스트:
{sample_text}

띄어쓰기 정확도 (0-100): """
        
        try:
            response = self._call_llm(prompt, temperature=0.1, max_tokens=100)
            
            # 숫자 추출
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0])
                return min(100.0, max(0.0, score))
            else:
                logger.warning("⚠️ LLM 띄어쓰기 점수 추출 실패, 기본값 사용")
                return 85.0
                
        except Exception as e:
            logger.warning(f"⚠️ 띄어쓰기 검사 실패: {str(e)}, 기본값 사용")
            return 85.0
    
    def _check_context_connection(self, content: str) -> float:
        """
        문맥 연결성 검사 (LLM 활용)
        """
        # 문단 경계 분석을 위한 샘플 추출
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 100.0
        
        # 샘플 문단들 추출 (최대 5개)
        sample_paragraphs = paragraphs[:5] if len(paragraphs) >= 5 else paragraphs
        sample_text = '\n\n'.join(sample_paragraphs)
        
        prompt = f"""한국어 문맥 연결 전문가로서, 다음 텍스트의 문맥 연결성을 0-100점으로 평가해주세요.

평가 기준:
- 문장 간 논리적 흐름
- 페이지 경계에서 끊어진 문장의 자연스러운 연결
- 전체적인 가독성과 일관성

텍스트:
{sample_text}

문맥 연결성 점수 (0-100): """
        
        try:
            response = self._call_llm(prompt, temperature=0.1, max_tokens=100)
            
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0])
                return min(100.0, max(0.0, score))
            else:
                return 80.0
                
        except Exception as e:
            logger.warning(f"⚠️ 문맥 연결성 검사 실패: {str(e)}, 기본값 사용")
            return 80.0
    
    def _check_structure_preservation(self, content: str) -> float:
        """
        구조 보존도 검사 (마크다운 구조 검증)
        """
        score = 100.0
        
        # 헤더 구조 검사
        headers = re.findall(r'^#+\s+.+$', content, re.MULTILINE)
        if headers:
            # 헤더가 적절히 구성되어 있는지 확인
            header_levels = [len(h.split()[0]) for h in headers]  # # 개수 세기
            
            # 헤더 레벨이 너무 급격히 변하지 않는지 확인
            for i in range(1, len(header_levels)):
                if header_levels[i] - header_levels[i-1] > 2:
                    score -= 5  # 급격한 헤더 레벨 변화 감점
        
        # 리스트 구조 검사
        list_items = re.findall(r'^\s*[-*+]\s+.+$', content, re.MULTILINE)
        numbered_items = re.findall(r'^\s*\d+\.\s+.+$', content, re.MULTILINE)
        
        # 기본적인 마크다운 구조가 있으면 가점
        if headers or list_items or numbered_items:
            score = min(100.0, score + 5)
        
        return max(70.0, score)  # 최소 70점 보장
    
    def _generate_improvements(self, sentence_score: float, spacing_score: float, 
                             context_score: float, structure_score: float) -> List[str]:
        """
        점수 기반 개선 제안 생성
        """
        improvements = []
        
        if sentence_score < 85:
            improvements.append("불완전한 문장들을 다음 문장과 연결하여 완성도를 높여주세요")
        
        if spacing_score < 85:
            improvements.append("띄어쓰기 오류를 수정하여 가독성을 개선해주세요")
        
        if context_score < 80:
            improvements.append("문맥 연결성을 강화하여 논리적 흐름을 개선해주세요")
        
        if structure_score < 80:
            improvements.append("마크다운 구조를 보완하여 문서 형태를 개선해주세요")
        
        if not improvements:
            improvements.append("품질이 우수합니다. 추가 개선 불필요")
        
        return improvements
    
    def is_quality_acceptable(self, quality_result: Dict[str, Any]) -> bool:
        """품질이 목표 수준에 도달했는지 확인"""
        return quality_result['passed']
    
    def get_target_score(self) -> float:
        """목표 점수 반환"""
        return self.target_score
    
    def set_target_score(self, score: float):
        """목표 점수 설정"""
        self.target_score = max(70.0, min(100.0, score))
        logger.info(f"🎯 목표 점수 설정: {self.target_score}")
