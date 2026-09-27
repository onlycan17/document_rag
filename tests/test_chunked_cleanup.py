"""분할 LLM 교정(chunked_cleanup) 테스트 — 가짜 LLM 사용"""

import re

import src.utils.chunked_cleanup as cleanup

SENTENCE = "몽촌토성 북문지 일원에서 백제 토기와 기와가 다수 출토되었다."
TWO_SENTENCES = f"{SENTENCE} {SENTENCE}"


def _section(prompt: str, name: str, next_name: str) -> str:
    return re.search(rf"\[{name}\]\n(.*?)\n\n\[{next_name}\]", prompt, re.S).group(1)


def _body(prompt: str) -> str:
    return _section(prompt, "본문", "뒤 문맥")


def test_chunks_never_end_mid_sentence(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 100)
    paragraphs = [TWO_SENTENCES, "끊긴 문장이 다음 문단으로 이어지는 경우의", TWO_SENTENCES, SENTENCE]
    chunks = cleanup.split_into_chunks("\n\n".join(paragraphs))

    assert "\n\n".join(chunks) == "\n\n".join(paragraphs)
    assert all(not chunk.endswith("경우의") for chunk in chunks)


def test_tables_are_not_split(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 30)
    table = "| 구분 | M | SD |\n| --- | --- | --- |\n| 사전 | 45.50 | 23.86 |\n| 사후 | 48.00 | 37.87 |"
    chunks = cleanup.split_into_chunks(f"{SENTENCE}\n\n{table}\n\n{SENTENCE}")

    assert any(table in chunk for chunk in chunks)


def test_oversized_run_is_cut_at_max_chars(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    monkeypatch.setattr(cleanup, "MAX_CHUNK_CHARS", 120)
    cut = "문장이 끝나지 않고 계속 이어지는 긴 문단의"
    chunks = cleanup.split_into_chunks("\n\n".join([cut] * 10))

    assert len(chunks) > 1 and all(len(chunk) <= 120 + len(cut) for chunk in chunks)


def test_neighbor_context_is_sent_but_only_body_is_kept(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return "[본문]\n" + _body(prompt).replace("출토되었다", "출토 되었다")

    text = "\n\n".join(f"{index}번 문단. {TWO_SENTENCES}" for index in range(3))  # 문단마다 한 청크
    result = cleanup.clean_document(text, fake_llm, max_workers=1)

    middle = next(p for p in prompts if "1번 문단" in _body(p))
    assert "0번 문단" in _section(middle, "앞 문맥", "본문")
    assert "2번 문단" in middle.split("[뒤 문맥]\n")[1]
    assert result.fallback_chunks == [] and result.text.count("출토 되었다") == 6 and "[본문]" not in result.text


def test_summarized_or_failed_chunks_fall_back_to_original(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    text = "\n\n".join(f"{index}번 문단. {TWO_SENTENCES}" for index in range(3))

    def fake_llm(prompt: str) -> str:
        body = _body(prompt)
        if body.startswith("0번"):
            return "요약: 토기 출토"  # 요약 → 길이 검증 실패
        if body.startswith("1번"):
            raise TimeoutError("응답 없음")
        return body

    result = cleanup.clean_document(text, fake_llm, max_workers=3)

    assert result.fallback_chunks == [1, 2]
    assert result.text == text


def test_llm_preambles_are_stripped():
    raw = "다음은 교정된 텍스트입니다:\n```\n본문 문장이다.\n```"
    assert cleanup.strip_llm_artifacts(raw) == "본문 문장이다."


def test_md_postprocessor_writes_cleaned_file_with_fallback_stats(tmp_path, monkeypatch):
    from src.utils.md_postprocessor import MDPostProcessor

    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    source = tmp_path / "doc.md"
    source.write_text("\n\n".join(f"{index}번 문단. {TWO_SENTENCES}" for index in range(2)), encoding="utf-8")
    processor = MDPostProcessor(output_dir=str(tmp_path / "out"), provider="openrouter", model_name="fake")
    monkeypatch.setattr(
        processor, "_call_llm", lambda prompt, **kwargs: "짧음" if _body(prompt).startswith("1번") else _body(prompt)
    )

    output = processor.process_md_file(str(source))

    assert (tmp_path / "out" / "doc_processed.md").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert output.endswith("doc_processed.md") and processor.stats["fallback_chunks"] == [2]


def test_context_connector_joins_pages_before_llm_cleanup(monkeypatch):
    from src.agents.context_connector import ContextConnectorAgent

    agent = ContextConnectorAgent(provider="openrouter", model_name="fake")
    bodies = []
    monkeypatch.setattr(agent, "_call_llm", lambda prompt, **kwargs: bodies.append(_body(prompt)) or _body(prompt))

    blocks = agent.process(
        ["한강이 서울지역을 강남과 강북으로 구분하는 경계이며 그 총", "연장은 497.5㎞이다.\n다음 문장이다."]
    )

    assert "그 총 연장은 497.5㎞이다." in bodies[0]
    assert blocks == ["한강이 서울지역을 강남과 강북으로 구분하는 경계이며 그 총 연장은 497.5㎞이다.\n\n다음 문장이다."]


def test_chunk_that_drops_page_marker_or_table_row_falls_back(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    text = "\n\n".join(
        [
            f"{TWO_SENTENCES}\n[페이지 1]\n{TWO_SENTENCES}",  # 청크 중간의 페이지 표시
            f"{TWO_SENTENCES}\n| 구분 | 값 |\n| --- | --- |\n| 사전 | 45.50 |",
            f"[페이지 3]\n{TWO_SENTENCES}",
        ]
    )

    def fake_llm(prompt: str) -> str:
        body = _body(prompt)
        if "[페이지 1]" in body:
            return body.replace("[페이지 1]\n", "")  # 페이지 표시 삭제
        if "| 사전 |" in body:
            return body.replace("| 사전 | 45.50 |", "사전 45.50")  # 표 행 변형
        return body.replace("| --- |", "|---|")  # 구분선 형식 변경은 허용

    result = cleanup.clean_document(text, fake_llm, max_workers=1)

    assert result.fallback_chunks == [1, 2]
    assert result.text == text


def test_failed_chunks_are_retried_once_sequentially(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    text = "\n\n".join(f"{index}번 문단. {TWO_SENTENCES}" for index in range(3))
    attempts = {}

    def flaky_llm(prompt: str) -> str:
        body = _body(prompt)
        attempts[body[:2]] = attempts.get(body[:2], 0) + 1
        if body.startswith("0번") and attempts["0번"] == 1:
            raise RuntimeError("429 Too Many Requests")  # 첫 시도만 실패
        if body.startswith("2번"):
            return "요약"  # 계속 실패
        return body

    result = cleanup.clean_document(text, flaky_llm, max_workers=3)

    assert attempts == {"0번": 2, "1번": 1, "2번": 2}
    assert result.fallback_chunks == [3]


def test_leading_page_marker_is_kept_out_of_llm_and_restored(monkeypatch):
    monkeypatch.setattr(cleanup, "TARGET_CHUNK_CHARS", 50)
    text = f"[페이지 74]\n{TWO_SENTENCES}\n\n[페이지 75]\n{TWO_SENTENCES}"
    bodies = []

    def llm_dropping_first_line_marker(prompt: str) -> str:
        body = _body(prompt)
        bodies.append(body)
        return re.sub(r"^\[페이지 \d+\]\n", "", body)  # 맨 앞 표시를 빠뜨리는 실제 패턴

    result = cleanup.clean_document(text, llm_dropping_first_line_marker, max_workers=1)

    assert not any(body.startswith("[페이지") for body in bodies)
    assert result.fallback_chunks == [] and result.text == text
