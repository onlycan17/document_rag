## 현재 작업
- [x] 문서와 코드 불일치 정리 및 수정
- [x] 이미지 분석 경로 OpenRouter 우선 적용 (ImageAnalyzer/Agent PDF Converter)
- [x] 로컬 LLM HTTP 폴백 비활성화로 인한 실패 해결(.env 정정)

## 리팩토링 패키지 진행 현황 (상세: `docs/refactor/TASKS.md`)
- [x] 서비스/스트리밍/전처리 구조 재정비
- [x] UI 스트리밍 및 이미지 렌더링 단순화
- [x] 전처리 파이프라인 공통화 및 RAG 메타데이터 정리

## 완료
- [x] Streamlit 리팩토링 패키지 실행
- [x] OCR 설치 경로 문서 수정 (`docs/INSTALL.md`, `docs/README_KOREAN.md`)
- [x] 존재하지 않는 스크립트 안내 제거 및 통합 스크립트로 교체 (`docs/UPSTAGE_SETUP_GUIDE.md`)
- [x] 업스테이지 임베딩 별도 설치 지침 추가 (`README.md`, `docs/INSTALL.md`, `docs/UPSTAGE_SETUP_GUIDE.md`)
- [x] 한국어 README의 저장소/경로/실행 방법 정정

## 다음 단계
- [ ] 개발 자동화 도입: `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`
- [ ] 네트워크 독립 스모크 테스트 추가(오프라인 실행 가능)
- [ ] `app.py` 모듈 분할 설계 및 리팩토링 계획 수립(<600 LOC 목표)
- [ ] CI 구성(GitHub Actions)으로 포맷/린트/테스트 강제
 - [ ] OpenRouter 이미지 분석 스로틀/재시도 로직 튜닝(429/타임아웃 대응)
 - [ ] 대용량 PDF(>150MB) 처리 시 진행률/에러 UI 개선
