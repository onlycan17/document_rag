"""_is_incomplete_sentence 캐릭터화 테스트 (골든 회귀).

현재 구현의 실제 출력 46케이스를 고정한다. sentence_completion.py로 분리한 뒤에도
동일 출력이 나와야 하며, 로직을 바꾸면 이 테스트가 실패한다(의도적 변경 시에만 갱신).
"""

import pytest

from src.utils.sentence_completion import is_incomplete_sentence

GOLDEN_CASES = [
        ('', False),
        ('   ', False),
        ('안녕하세요.', False),
        ('Really?', False),
        ('좋다!', False),
        ('합니다.', False),
        ('음.', False),
        ('였다.', False),
        ('다', True),
        ('음', True),
        ('그것은 중요하다', True),
        ('좋다', True),
        ('이 값은 확인되었다', True),
        ('구분하고 있', True),
        ('설치되고 있', True),
        ('놓고 있', True),
        ('만들어 있', True),
        ('확인하고', True),
        ('하며', True),
        ('하면서', True),
        ('건설되어', True),
        ('만드는', True),
        ('중요한', True),
        ('오는', True),
        ('만들', True),
        ('할', True),
        ('삼국사기 백', True),
        ('조선왕조실록 태', True),
        ('서울특별시 송', True),
        ('경기도 하', True),
        ('온조왕 태', True),
        ('475년 고', True),
        ('6세기 신', True),
        ('조선시대 전', True),
        ('자료 사', True),
        ('유물 발', True),
        ('보고서', True),
        ('그리고', True),
        ('또한', True),
        ('마지막으로', True),
        ('발굴흔적', True),
        ('유구', True),
        ('주요한', True),
        ('측정치를', True),
        ('자료의', True),
        ('현장에서', True),
        ('조사에', True),
        ('박물관에서', True),
        ('2024', True),
        ('Figure 3', True),
        ('abc', True),
        ('제1장', False),
        ('문화재 정 리', True),
        ('기단 석조물', True),
        ('조사하여,', True),
        ('발굴 하여', True),
        ('기록되며', True),
        ('완료 ㄹ', True),
        ('진행 ㅁ', True),
        ('This is a complete English sentence.', False),
        ('No period here friend', True),
]


@pytest.mark.parametrize("line,expected", GOLDEN_CASES)
def test_is_incomplete_sentence_golden(line, expected):
    assert is_incomplete_sentence(line) == expected
