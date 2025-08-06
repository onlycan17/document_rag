"""
품질 검증 에이전트 - 전처리 결과의 품질을 평가하고 개선 제안
"""

import re
import logging
from typing import Dict, List, Any, Tuple
from .base_agent import LocalLLMAgent

logger = logging.getLogger(__name__)


class QualityValidatorAgent(LocalLLMAgent):
    """
    전처리된 문서의 품질을 검증하고 문제점을 식별하는 에이전트
    """
    
    def __init__(self):
        super().__init__("QualityValidator")
        
        # 품질 검증 기준 예시들
        self.quality_examples = [
            "문맥 연결: '구분하고 있다' (좋음) vs '구분하고 있\\n다' (나쁨)",
            "제목 구조: '# 1. 조사개요' (좋음) vs '### 다.' (나쁨)",
            "고유명사: '삼국사기 백제본기' (좋음) vs '삼국사기 백\\n제본기' (나쁨)"
        ]
    
    def process(self, content: str) -> Dict[str, Any]:
        """
        문서 내용의 품질을 종합적으로 평가
        
        Args:
            content: 검증할 마크다운 내용
            
        Returns:
            품질 평가 결과 및 개선 제안
        """
        quality_report = {
            'overall_score': 0.0,
            'issues': [],
            'improvements': [],
            'metrics': {}
        }
        
        # 1. 문맥 연결 품질 검사
        context_score, context_issues = self._check_context_quality(content)
        quality_report['metrics']['context_score'] = context_score
        quality_report['issues'].extend(context_issues)
        
        # 2. 구조 품질 검사  
        structure_score, structure_issues = self._check_structure_quality(content)
        quality_report['metrics']['structure_score'] = structure_score
        quality_report['issues'].extend(structure_issues)
        
        # 3. 가독성 검사
        readability_score, readability_issues = self._check_readability(content)
        quality_report['metrics']['readability_score'] = readability_score
        quality_report['issues'].extend(readability_issues)
        
        # 4. 전체 점수 계산
        quality_report['overall_score'] = (context_score + structure_score + readability_score) / 3
        
        # 5. 개선 제안 생성
        if quality_report['overall_score'] < 0.8:
            quality_report['improvements'] = self._generate_improvements(quality_report)
        
        logger.info(f"📊 품질 검증 완료 - 전체 점수: {quality_report['overall_score']:.2f}")
        
        return quality_report
    
    def _check_context_quality(self, content: str) -> Tuple[float, List[str]]:
        """
        문맥 연결 품질 검사
        """
        issues = []
        
        # 1. 끊어진 문장 패턴 감지
        broken_patterns = [
            r'[가-힣]+\n###\s*[가-힣]',  # "있\n### 다" 패턴
            r'[가-힣]+\s*\n\s*[가-힣]{1,2}\.',  # 단어 중간 끊김
            r'『[^』]*\n[^』]*』',  # 책 제목 중간 끊김
        ]
        
        broken_count = 0
        for pattern in broken_patterns:
            matches = re.findall(pattern, content)
            broken_count += len(matches)
            for match in matches:
                issues.append(f"문맥 끊김 발견: '{match.replace(chr(10), '\\n')}'")
        
        # 2. LLM을 활용한 자연스러움 평가 (샘플링)
        sample_score = self._llm_evaluate_naturalness(content)
        
        # 점수 계산 (끊어진 패턴이 적을수록 높은 점수)
        total_lines = len(content.split('\n'))
        context_score = max(0.0, 1.0 - (broken_count / max(total_lines * 0.1, 1)))
        context_score = (context_score + sample_score) / 2
        
        return context_score, issues
    
    def _check_structure_quality(self, content: str) -> Tuple[float, List[str]]:
        """
        마크다운 구조 품질 검사
        """
        issues = []
        
        # 1. 잘못된 헤딩 패턴 검사
        wrong_headings = re.findall(r'^###\s*[가-힣]{1,2}\.\s*$', content, re.MULTILINE)
        for heading in wrong_headings:
            issues.append(f"잘못된 헤딩: '{heading}'")
        
        # 2. 헤딩 계층 구조 검사
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        structure_score = self._evaluate_heading_hierarchy(headings)
        
        # 3. 빈 헤딩 검사
        empty_headings = re.findall(r'^#{1,6}\s*$', content, re.MULTILINE)
        for empty in empty_headings:
            issues.append(f"빈 헤딩 발견: '{empty}'")
        
        final_score = max(0.0, structure_score - len(wrong_headings) * 0.1 - len(empty_headings) * 0.05)
        
        return final_score, issues
    
    def _check_readability(self, content: str) -> Tuple[float, List[str]]:
        """
        가독성 검사
        """
        issues = []
        
        # 1. 과도한 빈 줄 검사
        excessive_newlines = len(re.findall(r'\n{4,}', content))
        if excessive_newlines > 0:
            issues.append(f"과도한 빈 줄: {excessive_newlines}개 위치")
        
        # 2. 문장 길이 분포 검사
        sentences = re.split(r'[.!?]\s+', content)
        very_long_sentences = sum(1 for s in sentences if len(s) > 200)
        if very_long_sentences > len(sentences) * 0.2:
            issues.append(f"과도하게 긴 문장: {very_long_sentences}개")
        
        # 3. 특수문자 정리 상태 검사
        special_char_issues = len(re.findall(r'[^\w\s가-힣.,!?()[\]{}""''`#*-]', content))
        if special_char_issues > 20:
            issues.append(f"정리되지 않은 특수문자: {special_char_issues}개")
        
        # 점수 계산
        readability_score = max(0.0, 1.0 - len(issues) * 0.1)
        
        return readability_score, issues
    
    def _llm_evaluate_naturalness(self, content: str) -> float:
        """
        LLM을 활용한 자연스러움 평가 (샘플 기반)
        """
        # 내용에서 500자 정도의 샘플 추출
        sample = self._extract_sample(content, 500)
        
        prompt = self._create_korean_prompt(
            "이 텍스트의 자연스러움을 0-10점으로 평가",
            f"텍스트 샘플:\n{sample}",
            [
                "자연스러운 예시: '몽촌토성은 백제의 왕성으로 추정된다.' → 9점",
                "부자연스러운 예시: '몽촌토성은 백제의 왕\\n성으로 추정된다.' → 3점"
            ]
        )
        
        prompt += """

평가 기준:
- 문장의 완성도와 자연스러움
- 단어와 구문의 적절한 연결
- 전문 용어의 올바른 사용
- 문맥의 일관성

0-10점으로 점수만 응답하세요 (예: 8):
"""
        
        try:
            response = self._call_local_llm(prompt, temperature=0.1, max_tokens=10)
            score = float(re.search(r'\d+', response).group()) / 10.0
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"⚠️ 자연스러움 평가 실패: {str(e)}")
            return 0.7  # 기본값
    
    def _extract_sample(self, content: str, max_chars: int) -> str:
        """
        분석용 샘플 텍스트 추출
        """
        # 헤딩을 제외한 본문 내용만 추출
        lines = content.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        sample_text = '\n'.join(content_lines)
        return sample_text[:max_chars]
    
    def _evaluate_heading_hierarchy(self, headings: List[Tuple[str, str]]) -> float:
        """
        헤딩 계층 구조의 적절성 평가
        """
        if not headings:
            return 1.0
        
        # 헤딩 레벨 순서가 적절한지 검사
        prev_level = 0
        level_jumps = 0
        
        for heading_mark, title in headings:
            current_level = len(heading_mark)
            
            # 레벨이 2 이상 점프하는 경우
            if current_level > prev_level + 1:
                level_jumps += 1
            
            prev_level = current_level
        
        # 점수 계산 (레벨 점프가 적을수록 높은 점수)
        hierarchy_score = max(0.0, 1.0 - level_jumps * 0.2)
        return hierarchy_score
    
    def _generate_improvements(self, quality_report: Dict[str, Any]) -> List[str]:
        """
        품질 보고서를 바탕으로 개선 제안 생성
        """
        improvements = []
        
        if quality_report['metrics']['context_score'] < 0.7:
            improvements.append("문맥 연결 개선: ContextConnectorAgent 재실행 권장")
        
        if quality_report['metrics']['structure_score'] < 0.7:
            improvements.append("문서 구조 개선: 헤딩 계층 구조 재조정 필요")
        
        if quality_report['metrics']['readability_score'] < 0.7:
            improvements.append("가독성 개선: 문장 길이 조정 및 특수문자 정리 필요")
        
        return improvements