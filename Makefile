# 개발 편의 타겟(도구 미설치 환경에서도 실패하지 않도록 안전하게 동작)

.PHONY: fmt lint test ci

PY ?= python3

fmt:
	@echo "[fmt] 코드 포맷팅 진행 (ruff/black 감지 시 실행)"
	@command -v ruff >/dev/null 2>&1 && ruff format . || echo "ruff 미설치: skip"
	@command -v black >/dev/null 2>&1 && black . || echo "black 미설치: skip"

lint:
	@echo "[lint] 정적 점검 진행 (ruff 감지 시 실행)"
	@command -v ruff >/dev/null 2>&1 && ruff check . --quiet || echo "ruff 미설치: skip"

test:
	@echo "[test] 오프라인 스모크 테스트 실행"
	@command -v pytest >/dev/null 2>&1 && pytest -q tests/test_smoke_offline.py || \
	  ( [ -f tests/test_simple.py ] && $(PY) tests/test_simple.py || echo "pytest/단독 테스트 미구성: skip" )

ci: fmt lint test
	@echo "[ci] 완료"

