# UI 채팅 화면 개선 계획 (2026-09-12)

## 1. 배경
- 사용자 피드백: "채팅 화면이 부자연스럽다. UI를 더 신경 써 달라."
- 대상: ragTest Streamlit 챗봇 화면(`localhost:8502`, `app.py` + `ui/`).
- 실제 화면(Aside 브라우저)과 코드(`ui/components/chat_interface.py`, `src/utils/answer_formatter.py`)를 확인해 원인을 특정함.

## 2. 현황 문제 (확인됨)
1. **응답 마크다운 깨짐(핵심 원인)**: 화면에 `[출처 1, 2, 8, 9]### 참고 자료- 문서 1, ...` 처럼 `###` 헤딩이 문장 중간에 노출되고, `**옻(漆)**`가 굵게 처리되지 않음.
   - 원인: LLM 원문에서 헤딩이 앞 문장에 개행 없이 붙어 있고, `AnswerFormatter._match_section`이 **줄 단위**로만 헤딩을 인식 → 붙은 줄은 헤딩으로 인식되지 않고 본문에 그대로 남음.
2. **긴 답변이 한 덩어리**: `핵심 요약 / 상세 설명 / 참고 자료` 구분이 약하고 "상세 설명" 헤딩이 과도하게 큼.
3. **채팅처럼 안 보임**: `st.chat_message`를 써도 사용자/어시스턴트 구분이 약하고 본문이 전체 폭으로 퍼짐.
4. `[출처 N]`가 본문에 스타일 없이 그대로 박힘.
5. 컨트롤·메타 어색: `총 0개의 메시지` 캡션, `참조 문서 0개`/`모델 Unknown` 불일치.
6. 전체 톤: Streamlit 기본 테마 수준.

## 3. 이번 범위
- **1~3번만** 진행 (렌더링 정규화 · 답변 구조화 · 채팅 레이아웃/타이포).
- 4~6번(컨트롤/테마 토큰/메타 일관성)은 후속 작업으로 분리.

## 4. 설계
### 4.1 재사용 (신규 금지)
- 구조 생성은 이미 `src/utils/answer_formatter.AnswerFormatter`가 담당 → **새로 만들지 않는다.**
- 부족한 것은 (a) 붙은 헤딩 정규화, (b) 섹션 단위 **렌더링**, (c) 스타일.

### 4.2 신규 모듈 1 — `src/utils/answer_render.py` (순수 함수)
- `normalize_answer_markdown(text: str) -> str`
  - 앞 텍스트에 붙은 ATX 헤딩(`###`)을 줄 시작으로 분리: `…내용### 참고 자료` → `…내용\n\n### 참고 자료`
  - 헤딩 뒤에 바로 붙은 본문/불릿 분리: `### 참고 자료- 문서 1` → `### 참고 자료\n- 문서 1`
  - 3개 이상 연속 개행을 2개로 축약
- `split_answer_sections(text: str) -> list[AnswerSection]`
  - `AnswerFormatter`가 만든 정규 헤딩(`### 핵심 요약/상세 설명/참고 자료`)을 섹션으로 분해
  - 반환: `(kind, title, body)` 목록. 미인식 헤딩은 본문에 유지
- `format_citation_badges(text: str) -> str`
  - HTML 이스케이프(&, <, >) 후 `[출처 N]`, `[출처 N, M, ...]`를 `<span class="cite">출처 N</span>`로 치환
  - 이스케이프 → 삽입 순서를 지켜 HTML 주입 위험 차단

### 4.3 신규 모듈 2 — `ui/styles.py`
- `inject_custom_css()`: 기존 `app.py`의 이미지 라이트박스 CSS를 이관하고 아래 추가
  - `.cite` 인용 뱃지(작은 pill, 톤다운 색)
  - 채팅 메시지 내부 헤딩 스케일 축소(h1≈1.35rem, h2≈1.15rem, h3≈1.0rem), `line-height: 1.7`
  - `.answer-summary` 강조 박스(좌측 액센트 + 연한 배경)
  - `.stChatMessage` 간격, 본문 최대 폭 제한
- `app.py`의 인라인 `<style>`/`<script>`는 이관 후 `inject_custom_css()` 호출로 대체(라이트박스 동작 유지)

### 4.4 변경 — `ui/components/chat_interface.py`
- `render_assistant_content(text, documents)` 신설: 섹션을 파싱해
  - `핵심 요약` → `.answer-summary` 박스(마크다운 렌더)
  - `상세 설명` → `st.expander("상세 설명", expanded=True)`
  - `참고 자료` → 하단 카드(문서 정보 기반)
  - 미인식 → 정규화 마크다운
- **이력 렌더와 방금 생성한 응답이 같은 함수**를 쓰도록 통일(중복 제거).
- 스트리밍: 진행 중에는 기존대로 실시간 텍스트 표시, 완료 시 placeholder를 비우고 구조화 렌더로 교체(중복 출력 방지).

### 4.5 문서
- `docs/UI_UX_SPEC.md` 2·5절 갱신, `docs/CHANGELOG.md` 항목 추가, `docs/ARCHITECTURE.md`에 신규 모듈 반영.

## 5. TASK 목록
### 대분류 A. 응답 렌더링 정규화
- A-1 `src/utils/answer_render.py` 순수 함수 3종 구현
- A-2 `tests/test_answer_render.py` 작성(헤딩 분리/인용 뱃지/섹션 분해/HTML 이스케이프) — `tests/conftest.py`가 `utils/`를 network 마킹하므로 순수 오프라인 테스트는 루트에 둔다
- A-3 붙은 헤딩 케이스 회귀 테스트(실제 깨진 입력 재현)

### 대분류 B. 답변 구조화
- B-1 `chat_interface.render_assistant_content` 구현
- B-2 이력/live 렌더 통일, 참고 자료 카드화

### 대분류 C. 레이아웃/타이포
- C-1 `ui/styles.py` + `inject_custom_css()` 구현, `app.py` 이관
- C-2 버블 구분·헤딩 스케일·여백 CSS 적용

### 대분류 D. 검증/문서
- D-1 `python -m pytest tests/` + `make lint`(ruff) 통과
- D-2 Aside 브라우저로 개선 전/후 스크린샷 비교
- D-3 문서 동기화

## 6. 검증 기준
- 깨진 입력 재현 테스트 통과(헤딩이 본문에 남지 않음)
- 실제 질의 1회 → 화면에서 `###` 노출 없음, 요약 박스/상세 접기/참고 자료 카드 표시
- 오프라인 회귀 전체 통과, ruff 통과
