# OpenRouter 기반 PDF→Markdown 전처리/후처리 통합 설계서

목표(설명: 구현하고자 하는 바): PDF→MD 파이프라인에서 전처리(텍스트/멀티모달)와 후처리(MD 품질 개선)에 OpenRouter를 다른 제공자(OpenAI/Google/Anthropic)와 동일한 수준으로 선택·활용 가능하도록 확장한다.

## PRD(요구사항)
- 사용자(UI/설정)에서 전처리 모델로 `openrouter` 선택 가능
- 전처리 텍스트 모델 및 멀티모달 모델 선택 경로 제공
- MD 후처리에서도 `openrouter` 선택 및 키 사용 가능(이미 지원 중이지만 일관성 확보)
- 환경변수로 OpenRouter 모델/키를 설정하면 바로 동작
- 후크(포맷/린트/테스트) 모두 통과

## 아키텍처 개요
- Factory: `PreprocessingModelFactory`에 `openrouter` 제공자 추가
- API 모델: `APIPreprocessingModel`에 OpenRouter 분기 추가(텍스트 전처리)
- 멀티모달: `MultimodalPreprocessingModel`에 OpenRouter 비전 입력 처리 추가
- UI: `ui/components/sidebar.py` 전처리 모델 선택에 `openrouter` 노출, 키 체크/모델 표시 추가
- 설정: `config.py`에 OpenRouter 관련 키/모델 이미 존재 → 사용 및 주석 보강

## UI/UX 명세(요약)
- 전처리 모델 선택 드롭다운에 `openrouter` 추가
- 멀티모달 전처리 ON 시, OpenRouter 모델 후보를 표시(ENV 기반)
- API 키 존재 여부 시각화(초록/빨강 캡션)

## 기술 스택/ENV
- ENV 키: `OPNEROUTER_API_KEY`(주의: 철자 고정), `OPENROUTER_API_BASE`, `OPENROUTER_MODEL`, `OPENROUTER_MM_MODEL`
- 선택 확장: `EXTRA_MULTIMODAL_OPENROUTER_MODELS`

## 기능 명세
- 텍스트 전처리: Chat Completions 스타일 요청 → `choices[0].message.content`
- 멀티모달 전처리: 이미지 base64 data URL로 메시지 content에 첨부, OpenRouter Vision 모델 호출
- 실패 시 기존 로직과 동일하게 폴백(가능한 곳에서만)

## 테스트 계획
- 단위: Factory에 `openrouter` 추가로 인스턴스 생성 성공 여부
- 단위: APIPreprocessingModel의 provider=openrouter 분기 경로 호출(모킹)
- 단위: 멀티모달 OpenRouter 경로 메시지 생성 로직
- E2E 스모크: `python tests/test_simple.py` 정상 실행

## TODO
- [x] Factory: `openrouter` 제공자/기본모델/가용성/요구사항 추가
- [x] API 전처리 모델: OpenRouter 호출 분기 추가
- [x] 멀티모달 전처리: OpenRouter 처리 구현
- [x] UI: 전처리 모델 선택에 `openrouter` 추가 및 키 확인
- [x] 문서/README_KOREAN/주석 보강
- [x] make fmt && make test && make lint 통과

