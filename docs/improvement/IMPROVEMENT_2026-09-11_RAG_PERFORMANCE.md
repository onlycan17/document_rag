# RAG 성능 개선 회기 계획 (2026-09-11)

> **상태: 완료(2026-09-11)** — 임베딩 분리·평가 하네스·LangSmith 문서화·벡터 DB 재구축 완료.
> 평가 결과: Hit@5 100% / MRR 0.875 (10질의). 상세는 `docs/CHANGELOG.md` 참조.
> (2026-09-12) LangSmith는 관측 백엔드로 실제 전환 완료 — `docs/LANGSMITH_GUIDE.md`, CHANGELOG 2026-09-12 참조.

## 배경
프로젝트 구조 정리가 완료된 시점에서 검색 품질 개선으로 초점 이동. 현황 분석 결과:

1. **임베딩 모델 단일 사용**: 문서·질의 모두 `solar-embedding-1-large-query` 사용.
   업스테이지 권장은 문서 `-passage` / 질의 `-query` 분리 (검색 품질 저하 요인).
2. **평가 수단 부재**: 검색 품질 측정 도구(골든 셋, 지표)가 전혀 없어 튜닝 검증 불가.
3. **LangSmith 설정 무용화**: config에 키만 있고 어디서도 사용하지 않음.

## 범위 (이번 회기)

| 작업 | 내용 | 산출물 |
|---|---|---|
| 임베딩 모델 분리 | 문서 임베딩에 `solar-embedding-1-large-passage` 사용 (신규 `UPSTAGE_EMBEDDING_DOC_MODEL`) | embedding_model.py, config.py |
| LangSmith 트레이싱 | config의 4개 키를 os.environ에 반영해 langchain-core가 자동 수집하도록 최소 와이어링 | config.py |
| 검색 평가 하네스 | 골든 셋(JSON) + Hit@k/MRR 순수 지표 함수 + 실행 스크립트 | src/utils/retrieval_metrics.py, tests/eval/golden_retrieval.json, scripts/eval/retrieval_eval.py |

범위 제외 (근거):
- **LangGraph 미채택** — 검색→생성 선형 파이프라인에 그래프 오케스트레이션은 과설계
- **리랭커/RRF/Parent-Child 청킹** — 평가 하네스로 효과 측정 가능해진 뒤 후속 회기에서 판단
- **벡터 DB 재구축 자동 실행** — OCR·LLM 전처리 전체 파이프라인 재실행(API 비용·시간), 사용자 실행 필요

## 골든 셋 설계
- 근거: `data/documents/` 실재 문서 3종(몽촌토성4+상/하, KERIS LMM 연구)
- 판정 방식: 검색 결과 메타데이터 `source`에 기대 문서 파일명 포함 여부
- 지표: Hit@5, MRR (순수 함수로 분리해 오프라인 단위 테스트)

## 검증 절차
1. 오프라인 회귀망(ruff + pytest) 통과
2. `python scripts/eval/retrieval_eval.py` 실제 실행 — 현행(query 단일 모델) 기준 점수 확보
3. 벡터 DB 재구축 후 재실행해 passage 모델 효과를 숫자로 비교 (사용자 실행)

## 실행 결과 (2026-09-11)
- 벡터 DB 재구축 완료: converted_docs 마크다운 3종(몽촌토성4+상/하, KERIS) 기반 935 청크,
  문서 임베딩 passage 모델 적용. KERIS는 임베딩 429로 재구축에서 누락되어 증분 add_documents로 보강
- **기존 인덱스 문제 2건 발견·수정**: (1) 마지막 구축이 몽촌토성4+상.pdf만 인덱싱한 불완전 상태,
  (2) documents_cache.pkl 비어 있어 키워드 검색 무력화 → load_faiss_index에 docstore 복구 로직 추가
- 평가: 재구축 전 Hit@5 0%(정답 문서 미인덱싱 상태의 올바른 측정) → 재구축 후 **Hit@5 100%, MRR 0.875**
- LangSmith: `.env.example` 보호 파일 편집 차단으로 수동 추가 필요(아래 참조)

## 수동 추가 필요 (.env.example 보호 파일)
```
UPSTAGE_EMBEDDING_DOC_MODEL=solar-embedding-1-large-passage
LANGSMITH_TRACING=true
```
(LANGSMITH_API_KEY·ENDPOINT·PROJECT는 이미 존재)
