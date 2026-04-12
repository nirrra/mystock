from __future__ import annotations

from datetime import datetime

from stocks_analyzer.xueqiu_archive import (
    XUEQIU_USER_ID,
    XueqiuPost,
    _XueqiuArchiveService,
    extract_content_text,
    extract_post_id,
    extract_published_at,
    filter_post_urls,
)
from stocks_analyzer.xueqiu_rendering import render_xueqiu_archive_markdown


def test_filter_post_urls_keeps_only_target_user_posts() -> None:
    urls = [
        "/1155695148/379740650",
        "https://xueqiu.com/1155695148/379740650?foo=bar",
        "https://xueqiu.com/u/1155695148",
        "https://xueqiu.com/999999/123456",
        "https://example.com/1155695148/379740650",
        "https://xueqiu.com/1155695148/378975391",
    ]

    filtered = filter_post_urls(urls)

    assert filtered == [
        "https://xueqiu.com/1155695148/379740650",
        "https://xueqiu.com/1155695148/378975391",
    ]


def test_extract_post_id_returns_numeric_id() -> None:
    assert extract_post_id("https://xueqiu.com/1155695148/379740650") == "379740650"
    assert extract_post_id("https://xueqiu.com/u/1155695148") is None


def test_extract_published_at_normalizes_timestamp() -> None:
    text = "发布于2026-03-17 11:16来自雪球 · 广东"

    assert extract_published_at(text) == "2026-03-17 11:16"


def test_extract_content_text_removes_metadata_and_risk_notice() -> None:
    body_text = """
发布于2026-03-17 11:16来自雪球 · 广东
关注

#
来源：雪球App，作者： 老马盘股，（https://xueqiu.com/1155695148/379740650）

这么多创新药里面就$荣昌生物(SH688331)$ 比较符合预期，而且弹性也好。
三生制药也还行，$甘李药业(SH603087)$ 比较慢。
Image
01-13 12:23 · 讨论 84 · 赞 15

风险提示：用户发表的所有文章仅代表个人观点，与雪球的立场无关。
""".strip()

    content = extract_content_text(body_text)

    assert content == (
        "这么多创新药里面就$荣昌生物(SH688331)$ 比较符合预期，而且弹性也好。\n"
        "三生制药也还行，$甘李药业(SH603087)$ 比较慢。"
    )


def test_render_xueqiu_archive_markdown_sorts_descending() -> None:
    older = XueqiuPost(
        post_id="1",
        published_at="2026-03-10 09:30",
        url=f"https://xueqiu.com/{XUEQIU_USER_ID}/1",
        content_text="旧帖子",
    )
    newer = XueqiuPost(
        post_id="2",
        published_at="2026-03-11 10:30",
        url=f"https://xueqiu.com/{XUEQIU_USER_ID}/2",
        content_text="新帖子",
    )

    markdown = render_xueqiu_archive_markdown(
        [older, newer],
        generated_at=datetime(2026, 4, 11, 15, 0, 0),
        user_id=XUEQIU_USER_ID,
        candidate_count=2,
        failed_count=0,
    )

    assert markdown.index("## 2026-03-11 10:30") < markdown.index("## 2026-03-10 09:30")
    assert "候选链接数：2" in markdown
    assert "成功归档数：2" in markdown


def test_is_slider_verification_page_retries_after_navigation_context_reset() -> None:
    service = object.__new__(_XueqiuArchiveService)
    service.headed = False

    class FakePage:
        def __init__(self) -> None:
            self.url = "https://xueqiu.com/u/1155695148?alichlgref=https://xueqiu.com/u/1155695148"
            self._title_calls = 0

        def content(self) -> str:
            return "<html><title>滑动验证页面</title><body>访问验证 aliyun_waf</body></html>"

        def title(self) -> str:
            self._title_calls += 1
            if self._title_calls == 1:
                raise RuntimeError("Page.title: Execution context was destroyed, most likely because of a navigation")
            return "滑动验证页面"

        def wait_for_timeout(self, _milliseconds: int) -> None:
            return None

    assert service._is_slider_verification_page(FakePage()) is True
