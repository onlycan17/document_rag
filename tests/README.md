# 테스트 디렉토리 구조

RAG 시스템의 모든 테스트 코드를 관리하는 디렉토리입니다.

## 📁 디렉토리 구조

```
tests/
├── agent/           # 에이전트 관련 테스트
├── processing/      # 문서 처리 관련 테스트  
├── utils/          # 유틸리티 기능 테스트
├── integration/    # 통합 테스트
├── debug/          # 디버깅용 테스트
├── legacy/         # 레거시 테스트
└── outputs/        # 테스트 출력 파일 (Git 제외)
    ├── agent/      # 에이전트 테스트 출력
    ├── processing/ # 처리 테스트 출력
    └── logs/       # 테스트 로그 파일
```

## 🧪 카테고리별 테스트

### agent/ - 에이전트 테스트
- `test_agent_conversion.py`: 에이전트 기반 PDF 변환 테스트
- `test_agent_quick.py`: 빠른 에이전트 테스트
- `test_agent_simple.py`: 간단한 에이전트 기능 테스트

### processing/ - 처리 테스트
- `test_2stage_processing.py`: 2단계 MD 후처리 테스트
- `test_md_postprocessing.py`: MD 후처리 엔진 테스트
- `test_real_processing.py`: 실제 문서 처리 테스트

### utils/ - 유틸리티 테스트
- `test_clean_artifacts.py`: LLM 아티팩트 제거 테스트
- `test_pdf_conversion.py`: PDF 변환 기능 테스트
- `test_final_benchmark.py`: 최종 벤치마크 테스트
- `test_full_agent_system.py`: 전체 에이전트 시스템 테스트
- `test_small_sample.py`: 작은 샘플 테스트

### integration/ - 통합 테스트
전체 파이프라인 통합 테스트 (추가 예정)

### debug/ - 디버그 테스트
- `test_rag_context.py`: RAG 컨텍스트 디버깅
- `test_rag_query.py`: RAG 쿼리 디버깅

### legacy/ - 레거시 테스트
- `test_gpt41_rag.py`: GPT-4.1 RAG 테스트

### outputs/ - 테스트 출력 (Git 제외)
테스트 실행 시 생성되는 모든 출력 파일을 저장하는 디렉토리입니다.
- **agent/**: 에이전트 테스트의 변환 결과물
- **processing/**: 문서 처리 테스트의 중간 및 최종 결과물
- **logs/**: 테스트 실행 로그 파일
- ⚠️ 이 디렉토리는 `.gitignore`에 포함되어 버전 관리되지 않음
- 💡 주기적으로 정리하여 디스크 공간 관리 필요

## 🚀 테스트 실행 방법

### 개별 테스트 실행
```bash
# 가상환경 활성화
source venv/bin/activate

# 특정 테스트 실행
python tests/agent/test_agent_conversion.py
python tests/processing/test_2stage_processing.py
```

### 카테고리별 테스트 실행
```bash
# 모든 에이전트 테스트 실행
python -m pytest tests/agent/

# 모든 처리 테스트 실행
python -m pytest tests/processing/
```

### 전체 테스트 실행
```bash
# 모든 테스트 실행
python -m pytest tests/
```

## ✅ 테스트 작성 규칙

1. **파일명**: 모든 테스트 파일은 `test_*.py` 형식으로 작성
2. **위치**: 기능에 맞는 적절한 카테고리 디렉토리에 생성
3. **임포트**: 프로젝트 루트 기준으로 임포트 경로 설정
4. **격리**: 각 테스트는 독립적으로 실행 가능해야 함
5. **정리**: 테스트 후 생성된 파일은 자동 정리

## 📝 새 테스트 추가 템플릿

```python
#!/usr/bin/env python3
"""
테스트 설명
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_function():
    """테스트 함수"""
    # 테스트 로직
    pass

if __name__ == "__main__":
    test_function()
```

## ⚠️ 주의사항

- **프로젝트 루트에 테스트 파일 생성 금지**
- **프로젝트 루트에 test_* 디렉토리 생성 금지**
- 모든 테스트는 반드시 이 디렉토리 구조 내에서 관리
- 임시 테스트도 적절한 카테고리에 생성 (주로 debug/ 사용)
- 테스트 출력은 반드시 tests/outputs/ 디렉토리에 저장
- tests/outputs/는 주기적으로 정리하여 공간 관리