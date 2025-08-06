"""
MD 후처리 엔진 - 원본 MD 파일을 LLM으로 정제하여 완벽한 문맥 연결 MD 파일 생성
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.base_agent import LocalLLMAgent

logger = logging.getLogger(__name__)


class MDPostProcessor(LocalLLMAgent):
    """
    MD 파일 후처리 전문 클래스
    원본 MD → 문맥 연결된 고품질 MD 변환
    """
    
    def __init__(self, output_dir: str = "processed_docs", target_quality: int = 90):
        super().__init__("MDPostProcessor")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_quality = target_quality
        self.last_quality_score = 0.0
        self.quality_checker = None  # 품질 검사기 (옵션)
        
        # 처리 통계
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'total_chars': 0,
            'processing_time': 0.0
        }
        
        logger.info(f"🔧 MD 후처리 엔진 초기화 완료: {self.output_dir}")
    
    def set_quality_checker(self, quality_checker):
        """품질 검사기 설정"""
        self.quality_checker = quality_checker
        logger.info("품질 검사기 연결됨")
    
    def process(self, input_data: Any) -> Any:
        """
        에이전트 베이스 클래스의 추상 메서드 구현
        MD 파일 경로를 받아 처리된 파일 경로 반환
        """
        if isinstance(input_data, str):
            return self.process_md_file(input_data)
        else:
            raise ValueError("입력 데이터는 MD 파일 경로(문자열)여야 합니다")
    
    def process_file(self, md_file_path: str, progress_callback: Optional[Callable] = None) -> str:
        """
        MD 파일을 처리하는 메인 엔트리 포인트
        """
        return self.process_md_file(md_file_path, progress_callback)
    
    def process_md_file(self, md_file_path: str, progress_callback: Optional[Callable] = None) -> str:
        """
        MD 파일을 LLM으로 후처리하여 완벽한 문맥 연결 수행
        
        Args:
            md_file_path: 원본 MD 파일 경로
            progress_callback: 진행 상황 콜백 함수 (progress: float, message: str)
            
        Returns:
            처리된 MD 파일 경로
        """
        start_time = time.time()
        md_path = Path(md_file_path)
        
        if not md_path.exists():
            raise FileNotFoundError(f"MD 파일을 찾을 수 없습니다: {md_file_path}")
        
        logger.info(f"📝 MD 후처리 시작: {md_path.name}")
        
        if progress_callback:
            progress_callback(0.1, "MD 파일 읽기...")
        
        # 1단계: 원본 MD 파일 읽기
        with open(md_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        logger.info(f"📊 원본 텍스트: {len(original_content):,}자")
        
        if progress_callback:
            progress_callback(0.2, "텍스트 청킹...")
        
        # 2단계: 청크로 분할
        chunks = self._split_into_chunks(original_content)
        self.stats['total_chunks'] = len(chunks)
        self.stats['total_chars'] = len(original_content)
        
        logger.info(f"📋 {len(chunks)}개 청크로 분할")
        
        if progress_callback:
            progress_callback(0.3, f"{len(chunks)}개 청크 처리 시작...")
        
        # 3단계: 각 청크를 LLM으로 처리
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress = 0.3 + (i / len(chunks)) * 0.6
                progress_callback(progress, f"청크 {i+1}/{len(chunks)} 처리 중...")
            
            processed_chunk = self._process_chunk(chunk, i+1, len(chunks))
            processed_chunks.append(processed_chunk)
            self.stats['processed_chunks'] += 1
        
        if progress_callback:
            progress_callback(0.9, "처리된 텍스트 병합...")
        
        # 4단계: 처리된 청크들 병합
        final_content = self._merge_chunks(processed_chunks)
        
        # 5단계: 결과 저장
        output_filename = f"{md_path.stem}_processed.md"
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # 통계 업데이트
        self.stats['processing_time'] = time.time() - start_time
        
        # 품질 점수 계산 (옵션)
        if self.quality_checker:
            try:
                quality_result = self.quality_checker.analyze_quality(final_content)
                self.last_quality_score = quality_result.get('total_score', 0.0)
                logger.info(f"📊 품질 점수: {self.last_quality_score:.1f}점")
            except Exception as e:
                logger.warning(f"품질 검사 실패: {str(e)}")
                self.last_quality_score = 0.0
        
        if progress_callback:
            progress_callback(1.0, f"완료: {output_path.name}")
        
        logger.info(f"✅ MD 후처리 완료: {output_path}")
        logger.info(f"📊 처리 통계: {self.stats['processed_chunks']}/{self.stats['total_chunks']} 청크, "
                   f"{self.stats['processing_time']:.1f}초")
        
        return str(output_path)
    
    def _split_into_chunks(self, content: str, max_chunk_size: int = 2500) -> List[str]:
        """
        텍스트를 처리 가능한 청크로 분할 (문단 경계 고려)
        """
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = ""
        
        for line in lines:
            # 라인을 추가했을 때 크기 확인
            if len(current_chunk) + len(line) + 1 <= max_chunk_size:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _process_chunk(self, chunk: str, chunk_num: int, total_chunks: int) -> str:
        """
        개별 청크를 LLM으로 처리
        """
        prompt = f"""PDF에서 변환된 마크다운 텍스트의 품질을 개선해주세요.

규칙:
1. 문장 완성: '다.'로 끝나지 않는 모든 문장은 반드시 다음 텍스트와 연결
2. 띄어쓰기 교정: "운영에홍승연은" → "운영에 홍승연은"
3. 끊어진 단어 복원: "삼국사기 백제본기", "조선왕조실록 태조실록" 등
4. 제목/목차 구조 보존: 마크다운 헤더(#, ##, ###)와 리스트 구조 유지
5. 원본 내용 보존: 내용 추가/삭제 금지, 오직 연결과 교정만

⚠️ 매우 중요: 
- 어떠한 설명, 머리말, 도입부도 추가하지 마세요
- "한국어 문서 전문가로서..." 같은 역할 설명 금지
- "다음은 교정된 텍스트입니다" 같은 설명 금지
- 구분선(---)이나 메타 정보 추가 금지
- 오직 교정된 텍스트만 출력하세요

청크 {chunk_num}/{total_chunks}

입력:
{chunk}

출력(교정된 텍스트만):"""
        
        try:
            response = self._call_local_llm(prompt, temperature=0.1, max_tokens=3500)
            
            if response.strip():
                logger.debug(f"청크 {chunk_num} 처리 완료: {len(chunk)} → {len(response)}자")
                return response.strip()
            else:
                logger.warning(f"⚠️ 청크 {chunk_num} LLM 응답 없음, 원본 반환")
                return chunk
                
        except Exception as e:
            logger.warning(f"⚠️ 청크 {chunk_num} 처리 실패: {str(e)}, 원본 반환")
            return chunk
    
    def _merge_chunks(self, chunks: List[str]) -> str:
        """
        처리된 청크들을 자연스럽게 병합
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return self._clean_llm_artifacts(chunks[0])
        
        # 청크 간 경계에서 추가 연결 처리가 필요할 수 있음
        merged = chunks[0]
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            
            # 이전 청크 끝과 현재 청크 시작 분석
            prev_end = merged.strip()[-100:] if len(merged.strip()) > 100 else merged.strip()
            curr_start = current_chunk.strip()[:100] if len(current_chunk.strip()) > 100 else current_chunk.strip()
            
            # 청크 경계에서 문맥 연결이 필요한지 간단 판단
            if (prev_end and curr_start and 
                not prev_end.endswith('.') and 
                not prev_end.endswith('다.') and
                not curr_start.startswith('#')):  # 헤더가 아닌 경우
                
                # 자연스러운 연결
                merged += ' ' + current_chunk
            else:
                # 일반적인 병합 (줄바꿈 유지)
                merged += '\n\n' + current_chunk
        
        # 최종 결과에서 LLM 아티팩트 제거
        return self._clean_llm_artifacts(merged)
    
    def _clean_llm_artifacts(self, text: str) -> str:
        """
        LLM이 추가한 불필요한 설명이나 메타 텍스트 제거
        """
        import re
        
        # 제거할 패턴들
        patterns_to_remove = [
            # LLM 역할 설명
            r'한국어 문서 전문가로서[^\n]*\n?',
            r'다음은[^\n]*교정된[^\n]*텍스트[^\n]*:\n?',
            r'문맥을[^\n]*연결하겠습니다[^\n]*\n?',
            r'텍스트의 문맥을[^\n]*연결[^\n]*\n?',
            # 구분선과 메타 정보
            r'\n---+\s*한국어[^\n]*\n---+\n?',
            r'\n---+\s*$',  # 끝부분 구분선
            # 기타 설명
            r'^\s*##\s*한국문화사 문서에서[^\n]*\n?',
            # 추가 패턴들
            r'한국문화사 문서에서[^\n]*제공합니다[^\n]*\n?',
            r'문맥을[^\n]*연결하여[^\n]*제공합니다[^\n]*\n?',
            r'PDF에서 추출된[^\n]*연결하여[^\n]*\n?',
            r'^#+\s*한국어 문서[^\n]*제공합니다[^\n]*\n?',
            r'^---+\s*한국[^\n]*\n?',  # 시작부분 구분선
            r'^\s*---+\s*$',  # 단독 구분선
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # 연속된 줄바꿈 정리 (3개 이상 -> 2개)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # 앞뒤 공백 제거
        cleaned = cleaned.strip()
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'total_chars': 0,
            'processing_time': 0.0
        }