# 개발 편의 타겟 — 도구가 없으면 건너뛰고, 도구가 있는데 검사가 실패하면 실패로 끝낸다

.PHONY: fmt lint test ci

PY ?= python3

fmt:
	@echo "[fmt] 코드 포맷팅 진행 (ruff/black 감지 시 실행)"
	@if command -v ruff >/dev/null 2>&1; then ruff format .; else echo "ruff 미설치: skip"; fi
	@if command -v black >/dev/null 2>&1; then black .; else echo "black 미설치: skip"; fi

lint:
	@echo "[lint] 정적 점검 진행 (ruff 감지 시 실행)"
	@if command -v ruff >/dev/null 2>&1; then ruff check . --quiet; else echo "ruff 미설치: skip"; fi

test:
	@echo "[test] 오프라인 회귀망 실행"
	@if command -v pytest >/dev/null 2>&1; then pytest tests/ -q; \
	elif [ -f tests/test_simple.py ]; then $(PY) tests/test_simple.py; \
	else echo "pytest 미설치: skip"; fi

ci: fmt lint test
	@echo "[ci] 완료"
