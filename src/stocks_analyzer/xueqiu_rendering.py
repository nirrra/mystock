from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .xueqiu_archive import XueqiuPost


def render_xueqiu_archive_markdown(
    posts: Iterable["XueqiuPost"],
    *,
    generated_at: datetime,
    user_id: str,
    candidate_count: int,
    failed_count: int,
) -> str:
    ordered_posts = sorted(posts, key=lambda item: (item.published_at, item.post_id), reverse=True)
    lines = [
        f"# 雪球博主 {user_id} 历史帖子归档",
        "",
        "说明：",
        "本归档通过公开页面进行最佳努力抓取，不保证覆盖该账号全部历史帖子。",
        "",
        "统计：",
        f"- 生成时间：{generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 候选链接数：{candidate_count}",
        f"- 成功归档数：{len(ordered_posts)}",
        f"- 失败数：{failed_count}",
        "",
    ]
    for post in ordered_posts:
        lines.extend(
            [
                f"## {post.published_at}",
                "",
                f"原帖链接：[{post.url}]({post.url})",
                "",
                post.content_text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
