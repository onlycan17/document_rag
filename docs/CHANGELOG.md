# 변경 이력(Changelog)

## 2025-09-10
- feat(preprocessing): 기본 전처리 모델을 저비용·멀티모달 지향으로 재정렬
  - openai: gpt-4o-mini, google: gemini-1.5-flash-8b, anthropic: claude-3-5-haiku-20241022
- feat(multimodal): 멀티모달 전처리 기본 ON (ENABLE_MULTIMODAL_PREPROCESSING=true)
- feat(models): 멀티모달 지원 목록 최신화 + .env로 동적 확장(EXTRA_MULTIMODAL_*)
- feat(openrouter): 이미지 분석 기본 프로바이더를 OpenRouter로 전환, 모델 z-ai/glm-4.5v 설정
- fix(ui): app.py에 PreprocessingModelFactory 임포트 누락 보완
- feat(openai): Responses API 우선, 실패 시 Chat Completions 폴백

## 0.1.0 (초기 설계 문서화)
- PRD, 아키텍처, UI/UX, 기술 스택, 기능, API, ERD, 테이블, TODO/TASK 문서 최초 작성
- 코드베이스 역분석 기반 문서화 정합성 1차 확보
