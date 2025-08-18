#!/usr/bin/env python3
"""
A.X 멀티모달 모델 로딩 진행률 모니터링
"""

import time
import psutil
import subprocess
from pathlib import Path

def get_streamlit_process():
    """Streamlit 프로세스 찾기"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'streamlit' in proc.info['name'].lower():
                return proc
            if proc.info['cmdline'] and any('streamlit' in cmd for cmd in proc.info['cmdline']):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_memory_usage():
    """메모리 사용량 모니터링"""
    proc = get_streamlit_process()
    if not proc:
        print("❌ Streamlit 프로세스를 찾을 수 없습니다")
        return
    
    print(f"🔍 Streamlit 프로세스 모니터링 중 (PID: {proc.pid})")
    print("=" * 60)
    
    start_time = time.time()
    max_memory = 0
    
    try:
        while True:
            try:
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = proc.memory_percent()
                
                if memory_mb > max_memory:
                    max_memory = memory_mb
                
                elapsed = time.time() - start_time
                print(f"\r⏱️  {elapsed:.0f}초 | 🧠 {memory_mb:.1f}MB ({memory_percent:.1f}%) | 📈 최대: {max_memory:.1f}MB", end="", flush=True)
                
                # 메모리 사용량이 급증하면 모델 로딩 중
                if memory_mb > 3000:  # 3GB 이상
                    print(f"\n🚀 대용량 모델 로딩 중! 메모리: {memory_mb:.1f}MB")
                
                time.sleep(2)
                
            except psutil.NoSuchProcess:
                print(f"\n❌ 프로세스가 종료되었습니다 (최대 메모리: {max_memory:.1f}MB)")
                break
                
    except KeyboardInterrupt:
        print(f"\n🛑 모니터링 중단 (최대 메모리: {max_memory:.1f}MB)")

def check_log_progress():
    """로그 파일에서 진행률 확인"""
    log_file = Path("logs/rag_app_20250813.log")
    if not log_file.exists():
        print("❌ 로그 파일이 없습니다")
        return
    
    print("📋 최근 로그 엔트리:")
    result = subprocess.run(['tail', '-10', str(log_file)], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    print("🔄 A.X 멀티모달 모델 로딩 진행률 모니터링")
    print("=" * 60)
    
    check_log_progress()
    print("\n" + "=" * 60)
    monitor_memory_usage()