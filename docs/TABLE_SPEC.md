# 테이블(개념) 명세서

관계형 DB는 사용하지 않으나, 유지보수/확장 설계를 위해 개념적 스키마를 정의합니다.

## 1. documents (개념)
- `id`: string, 문서 식별자(파일명 기반 해시 권장)
- `file_name`: string, 실제 파일명
- `source_path`: string, 원본 경로
- `file_type`: enum(pdf|md|txt|etc)
- `extraction_method`: enum(pypdf|ocr|plain)
- `created_at`: datetime
- `metadata`: json

## 2. chunks (개념)
- `chunk_id`: string, `{file_name}_{index:04d}`
- `document_id`: string, FK → documents.id
- `text`: text
- `order`: int
- `metadata`: json

## 3. embeddings (개념)
- `chunk_id`: string, PK/FK → chunks.chunk_id
- `vector`: float[] (차원은 임베딩 모델에 따름)
- `model_name`: string
- `created_at`: datetime

## 4. vector_indexes (개념)
- `type`: enum(faiss|chroma)
- `path`: string, 인덱스 파일 위치
- `created_at`: datetime
- `meta`: json(파라미터/버전)

