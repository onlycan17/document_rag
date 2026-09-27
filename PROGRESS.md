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

## 다음 (3단계): 페이지 경계를 고려한 분할 전처리
- 문제: PDF 경로의 LLM 전처리(`preprocess_text`)가 문서 전체를 한 번에 보내 출력 4000토큰에서 잘림(35만 자 → 6천 자)
- 설계: ① `is_incomplete_sentence()`로 페이지 경계 규칙 연결 ② 완결 문장 문단 경계에서만 ~3000자 분할
  ③ 앞뒤 문맥을 참고용으로 함께 전달 ④ 출력 길이가 입력의 85~115% 밖이면 원문 사용 ⑤ 결과는 converted_docs/*.md로 검토 후 색인
- 대상: `src/agents/context_connector.py`(ContextConnectorAgent) 수정, `preprocess_text` 경로 대체

## 사용자 확인 필요
- `.env`의 `EXTRA_MULTIMODAL_*_MODELS`, `MD_POSTPROCESS_MODEL`에 옛 모델명 남아 있음
- Anthropic 계정 크레딧 부족(claude-haiku-4-5 호출 400)
- `vector_db_backup_20260927/` 백업 폴더 삭제 여부
