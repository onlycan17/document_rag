"""정크 텍스트 필터 테스트"""

import pytest
from langchain.schema import Document

from src.loaders.junk_filter import is_colophon, remove_junk_lines

KEEP_LINES = [
    "몽촌토성은 백제 한성기의 왕성으로 추정된다.",
    "9\tⅠ-13-28\t(16몽M-026)\t미상철기편\t철제\t5.7×5×1.8",  # 유물 목록 표 행
    "| 1 | 2 | 3 |",  # 마크다운 표 행
    "[페이지 35]",
    "2) Ⅰ-3호 수혈유구 조사 결과",
    "• 프로토타입의 긍정적인 점(P)",  # 글머리 기호는 유지
]

DROP_LINES = [
    "![이미지 1 (페이지 1)](images/대형 멀티모달 모델(LMMs) 적용 방안 연구(KERIS) F_page001_img001.png)",
    "![이미지](./images/몽촌토성4+상_page006_img001.png)",
    "□ 발간사·············································································5",
    "2) Ⅰ-3호 수혈유구················································80",
    "[그림 II-2] 대표적 멀티모달 모델 •••••",
    "11.6",
    "35",
    "이미지 1 (페이지 1) 적용 방안 연구(KERIS) Fpage001img001.png)",
    "이미지",
    "![image](/image/placeholder)",  # Upstage Document Parse 그림 자리표시자
    "**이미지**: 페이지 20의 이미지 ./images/몽촌토성4+하_page020_img006.png  ",
    "이미지: 페이지 10의 이미지 ./images/몽촌토성4+상page010img002.png",
    "### 페이지 내 이미지",
    "**이미지 설명**: No image provided; unable to perform document analysis or OCR.",
    "**이미지 설명**: No document image provided for analysis.",
    "**이미지 설명**: This is a completely black image with no visible content.",
    "**이미지 설명**: 이미지가 제공되지 않아 분석할 수 없습니다.",
    "**이미지 설명**: 입력에 이미지가 포함되지 않아 문서 이해와 관련된 분석이 불가능합니다.",
    "**이미지 설명**: 이미지를 확인할 수 없어 정확한 설명을 제공할 수 없습니다.",
]


@pytest.mark.parametrize("line", KEEP_LINES)
def test_meaningful_lines_are_kept(line):
    assert remove_junk_lines(line) == line


@pytest.mark.parametrize("line", DROP_LINES)
def test_junk_lines_are_removed(line):
    assert remove_junk_lines(line).strip() == ""


def test_meaningful_image_description_is_kept():
    line = "**이미지 설명**: 북문지 석축 전경 사진으로 성벽 기저부 석렬이 보인다."
    assert remove_junk_lines(line) == line


def test_wrapped_table_of_contents_entry_is_removed_entirely():
    toc = "사진 138 | Ⅰ-22호 수혈유구 출토유물 4 (20~22), Ⅰ-23호 수혈유구 (1 \n조사 전)··················148"
    assert remove_junk_lines(toc).strip() == ""


def test_body_caption_without_dot_leader_is_kept():
    text = "도면 110 | Ⅰ-25호 수혈유구 평ㆍ입단면도, 토층도\n25호 수혈유구는 해발 15.6m 지점에 위치한다."
    assert remove_junk_lines(text) == text


def test_image_with_descriptive_alt_keeps_alt_text():
    assert remove_junk_lines("![북문지 석축 전경](images/north(1).png)") == "북문지 석축 전경"


def test_model_box_tokens_are_stripped_but_content_kept():
    assert remove_junk_lines("<|begin_of_box|>토기편 출토<|end_of_box|>") == "토기편 출토"
    assert remove_junk_lines("<|beginofbox|>토기편") == "토기편"


def test_colophon_detection():
    assert is_colophon("Tel 02-2026-0545 ISBN 979-11-86012-88-8 Ⓒ 한성백제박물관, 2021")
    assert not is_colophon("ISBN이란 국제 표준 도서 번호를 말한다.")


def test_loader_preprocessing_removes_image_links_before_markdown_cleaning():
    from src.loaders.document_loader import EnhancedDocumentLoader

    loader = object.__new__(EnhancedDocumentLoader)
    loader._preprocessing_model = None
    body = "몽촌토성 북문지 일원에서 백제 토기와 기와가 다수 출토되었다. " * 3
    raw = f"{body}\n![이미지 1 (페이지 1)](images/연구(KERIS) F_page001_img001.png)\n<|begin_of_box|>{body}"

    content = loader._preprocess_documents([Document(page_content=raw)], ".md")[0].page_content

    assert ".png" not in content and "이미지" not in content and "box" not in content
    assert content.count("출토되었다") == 6
