# 테이블 명세서

관계형 데이터베이스 테이블을 직접 사용하지 않지만, 로그나 캐시 파일을 테이블 형태로 이해할 수 있는 구조로 정리합니다.

## 1. Document 메타데이터 (JSON/파케이티브 형태)
| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `file_name` | string | 원본 파일명 |
| `page` | int | 페이지 번호 |
| `image_paths` | list[string] | 추출된 이미지 경로 목록 |
| `processed_length` | int | 전처리 후 텍스트 길이 |

## 2. Chat 로그 (Streamlit 세션 상태 기반)
| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `role` | string | `user` 또는 `assistant` |
| `content` | string | 메시지 텍스트 |
| `context_documents` | list | 응답에 활용된 문서 정보 |
| `processing_time` | float | 응답 생성 시간(초) |

## 3. 벡터 스토어 엔트리 (FAISS/Chroma)
| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `id` | string | 청크 ID |
| `embedding` | vector | 벡터 데이터 |
| `metadata` | json | 파일명, 페이지, 태그 등 |
| `document` | string | 청크 텍스트 |

## 4. 용어 정리
- 메타데이터(설명: 데이터를 설명하는 보조 정보)
- 벡터(설명: 여러 숫자를 순서대로 나열한 것, 방향과 크기를 표현)
