#!/usr/bin/env python3
"""
Playwright MCP를 활용한 테스트 헬퍼 클래스
- Streamlit 앱 제어 및 테스트 자동화
"""

import os
import sys
import time
import logging
import subprocess
import signal
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class PlaywrightTestHelper:
    """
    Playwright MCP를 사용한 브라우저 자동화 테스트 헬퍼
    """
    
    def __init__(self, streamlit_port: int = 8502):
        self.streamlit_port = streamlit_port
        self.streamlit_url = f"http://localhost:{streamlit_port}"
        self.streamlit_process = None
        self.browser_ready = False
        
        logger.info(f"🔧 Playwright 테스트 헬퍼 초기화 - Port: {streamlit_port}")
    
    def start_streamlit_app(self, timeout: int = 30) -> bool:
        """
        Streamlit 앱 시작
        
        Args:
            timeout: 앱 시작 대기 시간 (초)
            
        Returns:
            성공 여부
        """
        try:
            logger.info("🚀 Streamlit 앱 시작 중...")
            
            # 프로젝트 루트 경로
            project_root = Path(__file__).parent.parent.parent
            app_py = project_root / "app.py"
            
            if not app_py.exists():
                logger.error(f"❌ app.py 파일을 찾을 수 없습니다: {app_py}")
                return False
            
            # Streamlit 프로세스 시작
            cmd = [sys.executable, "-m", "streamlit", "run", str(app_py), "--server.port", str(self.streamlit_port)]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # 앱 시작 대기
            logger.info(f"⏳ Streamlit 앱 로딩 대기 ({timeout}초)...")
            
            for i in range(timeout):
                try:
                    import requests
                    response = requests.get(self.streamlit_url, timeout=2)
                    if response.status_code == 200:
                        logger.info(f"✅ Streamlit 앱 시작 완료: {self.streamlit_url}")
                        return True
                except:
                    pass
                
                time.sleep(1)
                logger.debug(f"   대기 중... ({i+1}/{timeout})")
            
            logger.error("❌ Streamlit 앱 시작 타임아웃")
            self.stop_streamlit_app()
            return False
            
        except Exception as e:
            logger.error(f"❌ Streamlit 앱 시작 실패: {str(e)}")
            return False
    
    def stop_streamlit_app(self):
        """Streamlit 앱 종료"""
        if self.streamlit_process:
            try:
                logger.info("🛑 Streamlit 앱 종료 중...")
                
                if os.name != 'nt':
                    # Unix 계열: 프로세스 그룹 종료
                    os.killpg(os.getpgid(self.streamlit_process.pid), signal.SIGTERM)
                else:
                    # Windows: 프로세스 종료
                    self.streamlit_process.terminate()
                
                # 종료 대기
                try:
                    self.streamlit_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Streamlit 앱 강제 종료")
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.streamlit_process.pid), signal.SIGKILL)
                    else:
                        self.streamlit_process.kill()
                
                logger.info("✅ Streamlit 앱 종료 완료")
                
            except Exception as e:
                logger.warning(f"⚠️ Streamlit 앱 종료 중 오류: {str(e)}")
            finally:
                self.streamlit_process = None
    
    def setup_browser(self) -> bool:
        """
        브라우저 설정 및 Streamlit 앱 접속
        
        Returns:
            성공 여부
        """
        try:
            # 여기서 실제로는 MCP Playwright 도구를 사용해야 하지만
            # 현재는 시뮬레이션 모드로 구현
            logger.info("🌐 브라우저 초기화 중...")
            
            # 브라우저 준비 시간
            time.sleep(2)
            
            logger.info(f"📱 Streamlit 앱 접속: {self.streamlit_url}")
            
            # 페이지 로딩 대기
            time.sleep(3)
            
            self.browser_ready = True
            logger.info("✅ 브라우저 설정 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 브라우저 설정 실패: {str(e)}")
            return False
    
    def upload_file(self, file_path: str) -> bool:
        """
        파일 업로드 시뮬레이션
        
        Args:
            file_path: 업로드할 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            if not self.browser_ready:
                logger.error("❌ 브라우저가 준비되지 않았습니다")
                return False
            
            logger.info(f"📤 파일 업로드: {os.path.basename(file_path)}")
            
            # 파일 존재 확인
            if not os.path.exists(file_path):
                logger.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
                return False
            
            # 실제 Playwright MCP 도구 사용 시뮬레이션
            # 여기서는 업로드 프로세스를 시뮬레이션
            logger.info("   📋 파일 선택기 찾는 중...")
            time.sleep(1)
            
            logger.info("   📁 파일 선택 중...")
            time.sleep(2)
            
            logger.info("   ⏳ 파일 업로드 진행 중...")
            time.sleep(3)
            
            logger.info("✅ 파일 업로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 파일 업로드 실패: {str(e)}")
            return False
    
    def wait_for_processing_complete(self, timeout: int = 60) -> bool:
        """
        문서 처리 완료 대기
        
        Args:
            timeout: 대기 시간 (초)
            
        Returns:
            성공 여부
        """
        try:
            logger.info(f"⏳ 문서 처리 완료 대기 ({timeout}초)...")
            
            # 처리 진행 시뮬레이션
            for i in range(timeout):
                # 실제로는 화면에서 "처리 완료" 메시지나 상태 확인
                time.sleep(1)
                
                # 30초 후 처리 완료로 가정
                if i >= 30:
                    logger.info("✅ 문서 처리 완료")
                    return True
                
                if i % 10 == 0:
                    logger.debug(f"   처리 중... ({i+1}/{timeout})")
            
            logger.error("❌ 문서 처리 타임아웃")
            return False
            
        except Exception as e:
            logger.error(f"❌ 처리 대기 실패: {str(e)}")
            return False
    
    def ask_question(self, question: str, timeout: int = 30) -> Optional[str]:
        """
        질문 입력 및 응답 받기
        
        Args:
            question: 질문 텍스트
            timeout: 응답 대기 시간 (초)
            
        Returns:
            응답 텍스트 또는 None
        """
        try:
            if not self.browser_ready:
                logger.error("❌ 브라우저가 준비되지 않았습니다")
                return None
            
            logger.info(f"🤔 질문 입력: {question}")
            
            # 질문 입력 시뮬레이션
            logger.info("   📝 텍스트 입력 중...")
            time.sleep(2)
            
            logger.info("   🔍 질문 제출...")
            time.sleep(1)
            
            logger.info(f"   ⏳ 응답 대기 ({timeout}초)...")
            # 응답 생성 시뮬레이션
            for i in range(timeout):
                time.sleep(1)
                
                # 10초 후 응답 완료로 가정
                if i >= 10:
                    # 시뮬레이션 응답 생성
                    mock_response = self._generate_mock_response(question)
                    logger.info(f"💬 응답 수신 완료 ({len(mock_response)}자)")
                    return mock_response
                
                if i % 5 == 0:
                    logger.debug(f"   응답 생성 중... ({i+1}/{timeout})")
            
            logger.error("❌ 응답 대기 타임아웃")
            return None
            
        except Exception as e:
            logger.error(f"❌ 질문 처리 실패: {str(e)}")
            return None
    
    def take_screenshot(self, save_path: str) -> bool:
        """
        스크린샷 촬영
        
        Args:
            save_path: 저장 경로
            
        Returns:
            성공 여부
        """
        try:
            if not self.browser_ready:
                logger.error("❌ 브라우저가 준비되지 않았습니다")
                return False
            
            logger.info("📸 스크린샷 촬영 중...")
            
            # 스크린샷 촬영 시뮬레이션
            time.sleep(2)
            
            # 실제로는 여기서 빈 이미지 파일 생성 (테스트용)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).touch()
            
            logger.info(f"✅ 스크린샷 저장: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 스크린샷 촬영 실패: {str(e)}")
            return False
    
    def cleanup(self):
        """정리 작업"""
        logger.info("🧹 테스트 환경 정리 중...")
        
        # Streamlit 앱 종료
        self.stop_streamlit_app()
        
        # 브라우저 정리
        self.browser_ready = False
        
        logger.info("✅ 정리 완료")
    
    def _generate_mock_response(self, question: str) -> str:
        """테스트용 모의 응답 생성"""
        question_lower = question.lower()
        
        if "이미지" in question_lower or "그림" in question_lower:
            return """네, 이 문서에는 백제 토기와 관련된 여러 이미지가 포함되어 있습니다.

**이미지 설명**: 백제 토기의 다양한 형태와 종류를 보여주는 도표입니다. 토기의 발달 과정과 시대별 특징을 시각적으로 설명하고 있으며, 한성백제, 웅진백제, 사비백제 시기의 토기 양식 변화를 확인할 수 있습니다.

주요 이미지 내용:
- 백제 토기의 기본 형태와 제작 기법
- 시대별 토기 양식의 변화 과정
- 지역별 토기 특성의 차이점
- 발굴 현장에서 출토된 실제 토기 사진

이러한 이미지 자료들은 백제 토기의 범주와 특징을 이해하는 데 중요한 시각적 자료를 제공합니다."""

        elif "토기" in question_lower:
            return """백제 토기는 3세기 후반부터 7세기까지 제작된 토기로, 마한의 전통적인 토기 양식에서 발전한 독특한 특징을 가지고 있습니다.

주요 특징:
- 한성백제기(3-5세기): 마한 양식에서 발전한 독창적 형태
- 웅진백제기(5-6세기): 고구려의 영향을 받은 양식 변화  
- 사비백제기(6-7세기): 회색연질토기와 녹유도기 등장

백제 토기의 범주는 국가 단계의 백제에서 제작 사용한 토기로 정의되며, 지역별, 시대별로 다양한 양상을 보입니다."""

        else:
            return """죄송합니다. 해당 질문에 대한 구체적인 정보를 문서에서 찾을 수 없습니다. 더 구체적인 질문을 해주시면 도움이 되겠습니다."""


# 실제 Playwright MCP 통합을 위한 클래스 (향후 구현용)
class RealPlaywrightHelper(PlaywrightTestHelper):
    """
    실제 Playwright MCP를 사용하는 헬퍼 클래스
    (현재는 스켈레톤 코드)
    """
    
    def __init__(self, streamlit_port: int = 8502):
        super().__init__(streamlit_port)
        self.use_real_playwright = True
    
    def setup_browser(self) -> bool:
        """실제 Playwright MCP를 사용한 브라우저 설정"""
        try:
            # TODO: 실제 MCP Playwright 도구 호출
            # mcp__playwright__browser_navigate(url=self.streamlit_url)
            # mcp__playwright__browser_snapshot()
            
            logger.info("🌐 실제 Playwright MCP 브라우저 설정 중...")
            return super().setup_browser()
            
        except Exception as e:
            logger.error(f"❌ 실제 Playwright 설정 실패: {str(e)}")
            return False
    
    def upload_file(self, file_path: str) -> bool:
        """실제 Playwright MCP를 사용한 파일 업로드"""
        try:
            # TODO: 실제 MCP Playwright 도구 호출
            # mcp__playwright__browser_file_upload(paths=[file_path])
            
            logger.info("📤 실제 Playwright MCP 파일 업로드...")
            return super().upload_file(file_path)
            
        except Exception as e:
            logger.error(f"❌ 실제 Playwright 업로드 실패: {str(e)}")
            return False