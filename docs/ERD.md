# ERD(개념적 데이터 모델)

주의: 현재 프로젝트는 파일 시스템 기반(FAISS/Chroma)으로 동작하며, 관계형 DB는 사용하지 않습니다. 아래 ERD는 개념적 구조입니다(운영 설계/문서화를 위해 도식화).

## 1. 엔티티
- Document(원본 문서)
  - id, fileName, sourcePath, createdAt, metadata(json)
- Chunk(문서 청크)
  - chunkId, documentId, text, order, metadata(json)
- Embedding(청크 임베딩)
  - chunkId, vector(float[]), modelInfo
- VectorIndex(인덱스 스냅샷/백엔드별 파일)
  - type(FAISS/Chroma), path, createdAt

## 2. 관계(ASCII)

```
Document 1 ── * Chunk 1 ── 1 Embedding
                    │
                    └─ belongsTo → VectorIndex (물리적 저장은 파일/디렉토리)
```

## 3. 메타데이터 예시
- Document.metadata: {"file_type": "pdf", "extraction_method": "pypdf|ocr"}
- Chunk.metadata: {"chunk_id": "fileA_0005", "processing_method": "split_rules_v2"}

## 4. 보관 위치(실제 파일)
- `vector_db/` : 인덱스(`index.faiss`, `index.pkl`, 캐시 등)
- `processed_docs/` : 전처리 중간 산출물(옵션)

