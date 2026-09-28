# 진행 상황

브랜치: `fix/rag-embedding-mmr-metadata` (push 안 함)

## 완료
- 검색: MMR 재임베딩 제거·FAISS 거리 사용, 문서 캐시 누락 수정, 멀티모달 전처리 페이지 메타데이터 수정
- 정크 청크 필터(`src/loaders/junk_filter.py`) + 기존 인덱스 정리
- 스캔 PDF 판별 후 OCR 직행, OCR 엔진 모듈(`src/loaders/ocr_engines.py`): Upstage → Vision → Tesseract
- 비전·기본 모델 교체(qwen3.8-flash, gpt-6-luna, gemini-3.5-flash-lite, claude-haiku-4-5) + `settings.model_for` 통합
- 이미지 분석이 이미지를 모델에 보내지 않던 버그(`input_image`) 수정
- 공용 재시도가 5xx를 재시도하지 않던 문제 수정
- KERIS 보고서 Upstage 재변환·재색인 (인덱스 1028 청크)

- 3단계: 페이지 경계 규칙 연결 + 문장 경계 분할 LLM 교정(`chunked_cleanup`) + OCR 결과 검토 흐름 연결,
  전체 문서 LLM 전처리 제거 (KERIS: 59청크 중 52개 교정, 페이지 표시·표 보존)

- KERIS 교정본(97% 교정) 인덱스 반영, 실패 청크 재시도·앞머리 표시 보존·문맥 속 페이지 표시 제거

## 다음 후보
- 브랜치 `fix/rag-embedding-mmr-metadata` push·PR (아직 push 안 함)

## 사용자 확인 필요
- Anthropic 계정 크레딧 부족(claude-haiku-4-5 호출 400)
