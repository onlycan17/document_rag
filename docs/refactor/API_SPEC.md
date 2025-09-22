# API 명세서

이번 리팩토링은 외부 HTTP API를 새로 추가하지 않습니다. 다만 내부 서비스 간 호출 규칙을 정리합니다.

## 1. 내부 메서드 계약
| 제공자 | 메서드 | 입력 | 출력 | 설명 |
| --- | --- | --- | --- | --- |
| `StreamingResponseHandler` | `run(prompt: str)` | 사용자 질문 | `Dict`(답변, 소스, 메타데이터) | 스트리밍 응답을 수집하고 정리 |
| `ImageResolver` | `resolve(raw_path: str)` | 이미지 경로 후보 | 표준화된 절대 경로 또는 `None` | 이미지 경로 해석 |
| `ImageEncoder` | `encode(path: str)` | 절대 경로 | base64 URL 또는 `None` | 이미지 파일을 UI에서 쓸 수 있는 형태로 변환 |
| `DocumentPipeline` | `extract(file_path: str, **options)` | 파일 경로, 옵션 | `{text, metadata}` | 문서 추출과 전처리를 실행 |
| `LLMManager` | `get_model_metadata(provider: str, model: str)` | 공급자, 모델명 | `{max_tokens, context_window}` | 모델 메타데이터 제공 |

## 2. 예외 처리 규칙
- 이미지 경로 해석 실패 → `None` 반환, 호출 측에서 경고 표시.
- 문서 추출 실패 → 빈 텍스트와 오류 메타데이터 반환.
- 모델 메타데이터 없음 → 기본값을 `LLMManager`에서 관리.

## 3. 용어 정리
- 계약(설명: 함수를 사용할 때 입력과 출력이 어떤 모습이어야 하는지에 대한 약속)
- 메타데이터(설명: 데이터에 대한 추가 설명 정보)
