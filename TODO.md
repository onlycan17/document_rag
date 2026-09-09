## 현재 작업
- [x] 죽은 모듈 5종 삭제(약 2,500줄) + except-pass 에러 삼킴 17곳 로그화 (Phase 5)
- [x] CI/자동화: pyproject.toml(ruff) + GitHub Actions(ci.yml) + 오프라인 스모크 테스트 6종 (Phase 4)
- [x] requirements 정리: llama-cpp-python·timm·bitsandbytes 등 미사용 로컬 모델 의존성 제거, requirements-dev.txt 분리
- [x] 내장(로컬) 모델 호출 전면 제거 및 외부 API 전용 전환 (IMPROVEMENT_PLAN Phase 0)
- [x] 저장소 청소: app_original.py·디버그 스크립트 삭제, domain.md → docs/DOMAIN.md (Phase 1)
- [x] 로더 이중구조 통일: 레거시 생존 + refactored 삭제, 세션 초기화 잠재 버그 해소 (Phase 2)
- [x] 전처리 모델 콤보박스 제거 및 기본값 고정(사이드바 간소화)
- [x] 문서와 코드 불일치 정리 및 수정
- [x] 이미지 분석 경로 OpenRouter 우선 적용 (ImageAnalyzer/Agent PDF Converter)
- [x] 로컬 LLM HTTP 폴백 비활성화로 인한 실패 해결(.env 정정)
- [x] 로컬 LLM 재시도 설정 동기화 및 서킷 브레이커 도입(base_agent)
- [x] MD 후처리 OpenRouter 지원 추가(프로바이더/모델 오버라이드, base_agent 확장)
- [x] PDF→MD 전처리/후처리에서 OpenRouter를 제공자 옵션으로 추가(텍스트/멀티모달/UI)
- [x] 데이터 전처리 기본 제공자/모델 외부 API로 전환(로컬 1620 포트 호출 제거: config 기본값 수정)
 - [x] 에이전트 경로 강제 OpenRouter 고정(설정 `ENFORCE_OPENROUTER_FOR_AGENTS=true` 기본값, 모델 `z-ai/glm-4.5v`)

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
 - [ ] 레거시 document_loader.py 모듈 분할(600줄 가이드 준수, Phase 2 이연분)
 - [ ] 저장소 전체 ruff format 일괄 적용(별도 포매팅 커밋) 후 CI에 format --check 추가
 - [ ] 대용량 PDF(>150MB) 처리 시 진행률/에러 UI 개선
 - [ ] 로컬 LLM 장애 시 클라우드 폴백 옵션 도입(사용자 토글)

## 아키텍처 개선 작업 (새로운 섹션)
- [ ] 문서 처리 파이프라인 모듈화: `EnhancedDocumentLoader` 클래스 분리 (SRP 적용)
- [ ] 비동기 처리 도입: 문서 업로드 시 백그라운드 처리로 UI 블로킹 해결
- [ ] 캐시 계층 추가: Redis를 이용한 검색 결과 캐싱
- [ ] 마이크로서비스 아키텍처 설계: 문서 처리, 벡터 DB, 검색 서비스 분리
- [ ] API Gateway 도입 및 설정 관리 중앙화
- [ ] 모니터링 시스템 강화: 성능 메트릭스와 로깅 개선
- [ ] 분산 벡터 데이터베이스 구현: 여러 벡터 DB 샤드에 문서 분산 저장
- [ ] 머신러닝 기반 문서 품질 최적화 파이프라인 구축
- [ ] 실시간 스트리밍 아키텍처 도입: WebSocket을 이용한 문서 처리 상태 업데이트
