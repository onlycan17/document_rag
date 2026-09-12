"""앱 전역 커스텀 CSS 주입."""

from __future__ import annotations

import streamlit as st

_CUSTOM_CSS = """
<style>
:root {
    --chat-accent: #334e68;
    --chat-muted: #52606d;
    --chat-surface: #f3f6f8;
    --chat-border: #d9e2ec;
}
.main .block-container {
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}
[data-testid="stBottomBlockContainer"] { max-width: 960px; margin-inline: auto; }
.main h1 { font-size: 1.8rem; letter-spacing: -0.04em; }
[data-testid="stCaptionContainer"] { color: var(--chat-muted); }
[data-testid="stChatInput"] { border-radius: 16px; }
[data-testid="stChatInput"] textarea { min-height: 44px; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--chat-surface);
    border-radius: 16px;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: transparent;
    padding-top: 1.5rem;
}
[data-testid="stChatMessageContent"] { min-width: 0; overflow-wrap: anywhere; }
[data-testid="stChatMessage"] table { display: block; overflow-x: auto; }
@media (max-width: 640px) {
    .main .block-container { padding: 1.5rem 1rem 6rem; }
    [data-testid="stBottomBlockContainer"] { padding-inline: 1rem; }
    .main h1 { font-size: 1.5rem; }
    [data-testid="stChatMessage"] { padding: 0.75rem; }
}
/* 출처 인용 뱃지 */
.cite {
    display: inline-block;
    padding: 0 0.45em;
    margin: 0 0.15em;
    border-radius: 999px;
    background: var(--chat-surface);
    color: var(--chat-accent);
    font-size: 0.78em;
    font-weight: 600;
    line-height: 1.6;
    white-space: nowrap;
    vertical-align: baseline;
}

/* 채팅 답변 타이포: 과대 헤딩 방지 + 행간 확보 */
[data-testid="stChatMessage"] h1 { font-size: 1.35rem; margin: 0.4em 0 0.3em; }
[data-testid="stChatMessage"] h2 { font-size: 1.15rem; margin: 0.4em 0 0.3em; }
[data-testid="stChatMessage"] h3 { font-size: 1rem; margin: 0.35em 0 0.25em; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li { line-height: 1.7; }

/* 메시지 간격 */
[data-testid="stChatMessage"] { margin-bottom: 0.6rem; }

</style>
"""


def inject_custom_css() -> None:
    """커스텀 CSS를 앱에 주입한다."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
