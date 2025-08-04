#!/usr/bin/env python3
"""
RAG 챗봇 통합 실행 스크립트

이 스크립트는 RAG 챗봇 시스템의 모든 주요 기능을 메뉴 방식으로 
쉽게 접근할 수 있도록 제공합니다.

주요 기능:
1. Streamlit 앱 실행
2. 벡터 데이터베이스 관리
3. 테스트 실행
4. 환경 설정
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

def print_header() -> None:
    """
    헤더 출력
    
    RAG 챗봇 시스템의 제목과 구분선을 출력합니다.
    """
    print("=" * 60)
    print("🤖 RAG 챗봇 시스템 통합 실행 도구")
    print("=" * 60)
    print()

def print_menu() -> None:
    """
    메인 메뉴 출력
    
    사용 가능한 모든 기능의 메뉴를 출력합니다.
    """
    print("📋 사용 가능한 기능:")
    print()
    print("1️⃣  Streamlit 앱 실행")
    print("2️⃣  벡터 데이터베이스 관리")
    print("3️⃣  테스트 실행")
    print("4️⃣  환경 설정")
    print("5️⃣  프로젝트 정보")
    print("0️⃣  종료")
    print()

def run_streamlit_app():
    """Streamlit 앱 실행"""
    print("🚀 Streamlit 앱을 실행합니다...")
    print("브라우저에서 http://localhost:8501 에 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\n앱이 종료되었습니다.")
    except FileNotFoundError:
        print("❌ Streamlit이 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install streamlit")

def manage_vector_db():
    """벡터 데이터베이스 관리"""
    print("🗄️ 벡터 데이터베이스 관리")
    print("-" * 40)
    print()
    print("1. 전체 문서 처리 (기본)")
    print("2. 안전 모드 (OCR 비활성화)")
    print("3. 마크다운 파일만 처리")
    print("4. 고급 옵션 (사용자 정의)")
    print("0. 메인 메뉴로 돌아가기")
    print()
    
    choice = input("선택하세요 (0-4): ").strip()
    
    vector_script = PROJECT_ROOT / "scripts" / "data_management" / "vector_db_manager.py"
    
    if choice == "1":
        print("📚 전체 문서를 처리합니다...")
        subprocess.run([sys.executable, str(vector_script)])
    elif choice == "2":
        print("🛡️ 안전 모드로 처리합니다...")
        subprocess.run([sys.executable, str(vector_script), "--safe-mode"])
    elif choice == "3":
        print("📝 마크다운 파일만 처리합니다...")
        subprocess.run([sys.executable, str(vector_script), "--markdown-only"])
    elif choice == "4":
        print("🔧 고급 옵션:")
        print("사용 예시:")
        print(f"  python {vector_script} --help")
        print(f"  python {vector_script} --safe-mode --files \"file1.pdf\"")
        print(f"  python {vector_script} --types pdf txt")
        input("\nEnter를 눌러 계속하세요...")
    elif choice == "0":
        return
    else:
        print("❌ 잘못된 선택입니다.")

def run_tests():
    """테스트 실행"""
    print("🧪 테스트 실행")
    print("-" * 40)
    print()
    
    tests_dir = PROJECT_ROOT / "tests"
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ 테스트 파일을 찾을 수 없습니다.")
        return
    
    print("사용 가능한 테스트:")
    for i, test_file in enumerate(test_files, 1):
        test_name = test_file.stem.replace("test_", "").replace("_", " ").title()
        print(f"{i}. {test_name}")
    
    print("0. 모든 테스트 실행")
    print()
    
    try:
        choice = int(input("실행할 테스트를 선택하세요: ").strip())
        
        if choice == 0:
            print("🚀 모든 테스트를 실행합니다...")
            for test_file in test_files:
                print(f"\n▶️ {test_file.name} 실행 중...")
                subprocess.run([sys.executable, str(test_file)])
        elif 1 <= choice <= len(test_files):
            selected_test = test_files[choice - 1]
            print(f"🚀 {selected_test.name} 테스트를 실행합니다...")
            subprocess.run([sys.executable, str(selected_test)])
        else:
            print("❌ 잘못된 선택입니다.")
            
    except ValueError:
        print("❌ 숫자를 입력해주세요.")
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")

def setup_environment():
    """환경 설정"""
    print("⚙️ 환경 설정")
    print("-" * 40)
    print()
    print("1. 패키지 설치")
    print("2. OCR 설치")
    print("3. 환경 변수 설정 (.env)")
    print("4. 전체 설치 (패키지 + OCR)")
    print("0. 메인 메뉴로 돌아가기")
    print()
    
    choice = input("선택하세요 (0-4): ").strip()
    
    scripts_dir = PROJECT_ROOT / "scripts" / "setup"
    
    if choice == "1":
        print("📦 패키지를 설치합니다...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    elif choice == "2":
        print("🔍 OCR을 설치합니다...")
        ocr_script = scripts_dir / "install_ocr.sh"
        if ocr_script.exists():
            subprocess.run(["bash", str(ocr_script)])
        else:
            print("❌ OCR 설치 스크립트를 찾을 수 없습니다.")
    elif choice == "3":
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            print(f"📝 .env 파일이 존재합니다: {env_file}")
            print("기본 편집기로 열어서 API 키를 설정하세요.")
        else:
            print("📝 .env 파일을 생성합니다...")
            # 기본 .env 파일 생성
            env_content = """# RAG 챗봇 환경 변수 설정

# 업스테이지 API 키 (필수)
UPSTAGE_API_KEY=your_upstage_api_key_here

# OpenAI API 키 (선택)
OPENAI_API_KEY=your_openai_api_key_here

# Google API 키 (선택)
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic API 키 (선택)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 로컬 LLM 설정 (선택)
OLLAMA_BASE_URL=http://localhost:11434

# 기타 설정
ANONYMIZED_TELEMETRY=False
CHROMA_TELEMETRY=False
"""
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_content)
            print(f"✅ .env 파일을 생성했습니다: {env_file}")
            print("파일을 열어서 실제 API 키로 수정해주세요.")
    elif choice == "4":
        print("🔧 전체 설치를 진행합니다...")
        install_script = scripts_dir / "install_and_run.sh"
        if install_script.exists():
            subprocess.run(["bash", str(install_script)])
        else:
            print("❌ 설치 스크립트를 찾을 수 없습니다.")
    elif choice == "0":
        return
    else:
        print("❌ 잘못된 선택입니다.")

def show_project_info():
    """프로젝트 정보 출력"""
    print("ℹ️ 프로젝트 정보")
    print("-" * 40)
    print()
    
    # 프로젝트 구조 표시
    print("📁 프로젝트 구조:")
    print(f"  📂 {PROJECT_ROOT}")
    print("  ├── 📄 app.py              (메인 앱)")
    print("  ├── 📄 config.py           (설정)")
    print("  ├── 📄 requirements.txt    (의존성)")
    print("  ├── 📂 src/               (소스 코드)")
    print("  ├── 📂 scripts/           (실행 스크립트)")
    print("  │   ├── 📂 setup/         (설치/설정)")
    print("  │   └── 📂 data_management/ (데이터 관리)")
    print("  ├── 📂 tests/             (테스트)")
    print("  ├── 📂 docs/              (문서)")
    print("  ├── 📂 data/              (데이터)")
    print("  └── 📂 logs/              (로그)")
    print()
    
    # 주요 파일 상태 확인
    print("📊 시스템 상태:")
    
    # .env 파일 확인
    env_file = PROJECT_ROOT / ".env"
    env_status = "✅ 존재" if env_file.exists() else "❌ 없음"
    print(f"  .env 파일:        {env_status}")
    
    # 벡터 DB 확인
    vector_db_dir = PROJECT_ROOT / "vector_db"
    vector_status = "✅ 존재" if vector_db_dir.exists() else "❌ 없음"
    print(f"  벡터 DB:          {vector_status}")
    
    # 문서 디렉토리 확인
    docs_dir = PROJECT_ROOT / "data" / "documents"
    if docs_dir.exists():
        doc_count = len(list(docs_dir.glob("*.*")))
        print(f"  문서 파일:        ✅ {doc_count}개")
    else:
        print(f"  문서 파일:        ❌ 디렉토리 없음")
    
    print()
    print("📚 도움말:")
    print("  • 처음 사용: 4번 환경 설정부터 시작하세요")
    print("  • 문서 추가: data/documents/ 폴더에 파일 복사")
    print("  • 벡터 DB 재구축: 2번 메뉴 사용")
    print("  • 앱 실행: 1번 메뉴 사용")
    print()
    
    input("Enter를 눌러 계속하세요...")

def main():
    """메인 함수"""
    try:
        while True:
            print_header()
            print_menu()
            
            choice = input("메뉴를 선택하세요 (0-5): ").strip()
            print()
            
            if choice == "1":
                run_streamlit_app()
            elif choice == "2":
                manage_vector_db()
            elif choice == "3":
                run_tests()
            elif choice == "4":
                setup_environment()
            elif choice == "5":
                show_project_info()
            elif choice == "0":
                print("👋 종료합니다. 감사합니다!")
                break
            else:
                print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                input("Enter를 눌러 계속하세요...")
            
            print()
            
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 