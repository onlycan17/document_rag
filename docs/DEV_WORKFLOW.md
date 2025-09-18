# 개발 워크플로우

## 기본 명령
- 포맷팅: `make fmt` (ruff/black 감지 시 자동 실행)
- 린트: `make lint` (ruff 감지 시 실행)
- 테스트: `make test` (pytest 설치 시 실행, 없으면 스모크 대체)
- 전체: `make ci`

도구가 미설치된 환경에서도 실패하지 않도록 안전하게 동작합니다. 정식 훅은 아래 pre-commit을 권장합니다.

## pre-commit 훅
```bash
pip install pre-commit
pre-commit install
```
- 적용 훅: 공백/EOF/병합충돌/파일크기/형식 체크 + ruff/ruff-format

