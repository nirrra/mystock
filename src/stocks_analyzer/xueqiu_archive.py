from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

from .paths import ProjectPaths
from .xueqiu_rendering import render_xueqiu_archive_markdown

LOGGER = logging.getLogger(__name__)

XUEQIU_USER_ID = "1155695148"
XUEQIU_PROFILE_URL = f"https://xueqiu.com/u/{XUEQIU_USER_ID}"
_XUEQIU_POST_URL_RE = re.compile(r"^/(\d+)/(\d+)(?:/)?$")
_PUBLISHED_AT_RE = re.compile(r"(?:发布于|修改于)\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)")
_ENGAGEMENT_LINE_RE = re.compile(r"^\d{2}-\d{2}\s+\d{2}:\d{2}\s+·")
_TITLE_LINE_RE = re.compile(r"^#\s*\S")
_META_PREFIXES = (
    "发布于",
    "修改于",
    "关注",
    "来源：雪球App",
    "风险提示：",
)
_DETAIL_TEXT_SELECTORS = (
    "article",
    "main article",
    "[class*='article']",
    "[class*='status-content']",
    "[class*='detail']",
    "[class*='content']",
    "main",
    "body",
)


@dataclass(slots=True, frozen=True)
class XueqiuPost:
    post_id: str
    published_at: str
    url: str
    content_text: str


@dataclass(slots=True, frozen=True)
class XueqiuArchiveResult:
    output_path: Path
    candidate_count: int
    archived_count: int
    failed_count: int
    used_cache: bool


def archive_xueqiu_user_1155695148(
    paths: ProjectPaths,
    *,
    output: str | None = None,
    max_posts: int | None = None,
    refresh: bool = False,
    headed: bool = False,
) -> XueqiuArchiveResult:
    service = _XueqiuArchiveService(paths, headed=headed)
    return service.run(output=output, max_posts=max_posts, refresh=refresh)


def filter_post_urls(urls: Iterable[str], *, user_id: str = XUEQIU_USER_ID) -> list[str]:
    normalized_urls: list[str] = []
    seen: set[str] = set()
    for raw_url in urls:
        if not raw_url:
            continue
        parsed = urlparse(urljoin("https://xueqiu.com", raw_url.strip()))
        if parsed.netloc not in {"xueqiu.com", "www.xueqiu.com"}:
            continue
        match = _XUEQIU_POST_URL_RE.match(parsed.path)
        if not match or match.group(1) != user_id:
            continue
        normalized = f"https://xueqiu.com/{user_id}/{match.group(2)}"
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_urls.append(normalized)
    return normalized_urls


def extract_post_id(url: str) -> str | None:
    match = _XUEQIU_POST_URL_RE.match(urlparse(url).path)
    if match:
        return match.group(2)
    return None


def extract_published_at(text: str) -> str | None:
    match = _PUBLISHED_AT_RE.search(text)
    if not match:
        return None
    value = match.group(1)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d %H:%M")
    return value


def extract_content_text(text: str) -> str | None:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    lines = [line.strip() for line in cleaned.split("\n")]

    content_started = False
    result_lines: list[str] = []
    for line in lines:
        if not line:
            if result_lines and result_lines[-1] != "":
                result_lines.append("")
            continue
        if line.startswith("风险提示："):
            break
        if line == "Image" or _ENGAGEMENT_LINE_RE.match(line):
            continue
        if not content_started and line.startswith("来源：雪球App"):
            content_started = True
            continue
        if not content_started:
            if any(line.startswith(prefix) for prefix in _META_PREFIXES):
                continue
            if line == "#":
                continue
            if _TITLE_LINE_RE.match(line):
                content_started = True
            elif result_lines:
                content_started = True
            else:
                content_started = True
        if line in {"评论", "转发", "赞赏"}:
            continue
        result_lines.append(line)

    while result_lines and result_lines[-1] == "":
        result_lines.pop()
    if not result_lines:
        return None
    return "\n".join(result_lines).strip()


class _XueqiuArchiveService:
    def __init__(self, paths: ProjectPaths, *, headed: bool) -> None:
        self.paths = paths
        self.headed = headed
        self.cache_dir = self.paths.base_data_dir / "xueqiu" / XUEQIU_USER_ID
        self.raw_dir = self.cache_dir / "raw"
        self.browser_profile_dir = self.cache_dir / "browser_profile"
        self.discovered_urls_path = self.cache_dir / "discovered_urls.json"
        self.output_dir = self.paths.reports_dir / "xueqiu"

    def run(
        self,
        *,
        output: str | None,
        max_posts: int | None,
        refresh: bool,
    ) -> XueqiuArchiveResult:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.browser_profile_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        discovered_urls, used_cache = self._load_or_discover_urls(refresh=refresh)
        if max_posts is not None:
            discovered_urls = discovered_urls[:max_posts]
        if not discovered_urls:
            raise RuntimeError("No candidate Xueqiu post URLs were discovered.")

        posts: list[XueqiuPost] = []
        failed_count = 0
        with self._playwright_context() as context:
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(15_000)
            for url in discovered_urls:
                try:
                    post = self._extract_post(page, url)
                except Exception as exc:  # pragma: no cover - browser/runtime dependent
                    failed_count += 1
                    LOGGER.warning("Failed to archive %s: %s", url, exc)
                    continue
                if post is None:
                    failed_count += 1
                    LOGGER.warning("Skipped %s because published time or content was missing", url)
                    continue
                posts.append(post)
                self._write_raw_post(post)

        deduped_posts = {post.post_id: post for post in posts}
        if not deduped_posts:
            raise RuntimeError("No Xueqiu posts were archived successfully.")

        output_path = Path(output).resolve() if output else (self.output_dir / f"{XUEQIU_USER_ID}.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = render_xueqiu_archive_markdown(
            deduped_posts.values(),
            generated_at=datetime.now(),
            user_id=XUEQIU_USER_ID,
            candidate_count=len(discovered_urls),
            failed_count=failed_count,
        )
        output_path.write_text(markdown, encoding="utf-8")
        return XueqiuArchiveResult(
            output_path=output_path,
            candidate_count=len(discovered_urls),
            archived_count=len(deduped_posts),
            failed_count=failed_count,
            used_cache=used_cache,
        )

    def _load_or_discover_urls(self, *, refresh: bool) -> tuple[list[str], bool]:
        if not refresh and self.discovered_urls_path.exists():
            cached_urls = json.loads(self.discovered_urls_path.read_text(encoding="utf-8"))
            return filter_post_urls(cached_urls), True

        with self._playwright_context() as context:
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(15_000)
            urls = self._discover_urls(page)
        self.discovered_urls_path.write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8")
        return urls, False

    def _discover_urls(self, page) -> list[str]:
        self._goto(page, XUEQIU_PROFILE_URL)
        discovered_urls: list[str] = []
        stagnant_rounds = 0
        for _ in range(18):
            hrefs = page.eval_on_selector_all("a[href]", "nodes => nodes.map(node => node.href)")
            filtered = filter_post_urls(hrefs)
            if len(filtered) > len(discovered_urls):
                discovered_urls = filtered
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            if stagnant_rounds >= 4:
                break
            page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            page.wait_for_timeout(1_000)
        return discovered_urls

    def _extract_post(self, page, url: str) -> XueqiuPost | None:
        self._goto(page, url)
        text = self._extract_best_detail_text(page)
        published_at = extract_published_at(text)
        content_text = extract_content_text(text)
        post_id = extract_post_id(url)
        if not post_id or not published_at or not content_text:
            return None
        return XueqiuPost(
            post_id=post_id,
            published_at=published_at,
            url=url,
            content_text=content_text,
        )

    def _extract_best_detail_text(self, page) -> str:
        candidates: list[str] = []
        for selector in _DETAIL_TEXT_SELECTORS:
            try:
                text = page.locator(selector).first.inner_text(timeout=2_000).strip()
            except Exception:  # pragma: no cover - browser/runtime dependent
                continue
            if text:
                candidates.append(text)
        if not candidates:
            return ""
        return max(candidates, key=len)

    def _goto(self, page, url: str) -> None:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(3_000)
        self._handle_slider_verification(page)

    def _write_raw_post(self, post: XueqiuPost) -> None:
        raw_path = self.raw_dir / f"{post.post_id}.json"
        raw_path.write_text(
            json.dumps(
                {
                    "post_id": post.post_id,
                    "published_at": post.published_at,
                    "url": post.url,
                    "content_text": post.content_text,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _playwright_context(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(
                "xueqiu-archive requires playwright. Run `pip install playwright` and ensure Microsoft Edge is installed."
            ) from exc

        playwright_manager = sync_playwright().start()
        proxy_env_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        saved_proxy_env = {name: os.environ.get(name) for name in proxy_env_names}
        try:
            for name in proxy_env_names:
                os.environ.pop(name, None)
            try:
                context = playwright_manager.chromium.launch_persistent_context(
                    user_data_dir=str(self.browser_profile_dir),
                    channel="msedge",
                    headless=not self.headed,
                    locale="zh-CN",
                    viewport={"width": 1440, "height": 2400},
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ],
                )
            except Exception:  # pragma: no cover - environment dependent
                context = playwright_manager.chromium.launch_persistent_context(
                    user_data_dir=str(self.browser_profile_dir),
                    headless=not self.headed,
                    locale="zh-CN",
                    viewport={"width": 1440, "height": 2400},
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ],
                )
            context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en-US', 'en'] });
                Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                window.chrome = window.chrome || { runtime: {} };
                """
            )
            LOGGER.info("Using Xueqiu browser profile at %s", self.browser_profile_dir)
            return _ManagedBrowserContext(playwright_manager, context)
        except Exception:
            playwright_manager.stop()
            raise
        finally:
            for name, value in saved_proxy_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _handle_slider_verification(self, page) -> None:
        if not self._is_slider_verification_page(page):
            return
        if not self.headed:
            raise RuntimeError("Xueqiu returned a slider verification page. Re-run with --headed and complete the browser verification.")

        LOGGER.warning("Xueqiu requires slider verification. Complete it in the opened browser window; waiting up to 90 seconds.")
        for _ in range(45):
            page.wait_for_timeout(2_000)
            if not self._is_slider_verification_page(page):
                return
        raise RuntimeError("Timed out waiting for manual slider verification to complete.")

    def _is_slider_verification_page(self, page) -> bool:
        for _ in range(5):
            try:
                current_url = page.url
                page_content = page.content()
                page_title = page.title()
            except Exception as exc:  # pragma: no cover - browser/runtime dependent
                if "Execution context was destroyed" in str(exc) or "Most likely the page has been closed" in str(exc):
                    page.wait_for_timeout(500)
                    continue
                raise

            if "alichlgref=" in current_url:
                return True
            if page_title == "滑动验证页面":
                return True
            if "aliyun_waf" in page_content and ("滑动验证" in page_content or "_waf_" in page_content):
                return True
            return False

        return "alichlgref=" in page.url


class _ManagedBrowserContext:
    def __init__(self, playwright_manager, context) -> None:
        self.playwright_manager = playwright_manager
        self.context = context

    def __enter__(self):
        return self.context

    def __exit__(self, exc_type, exc, tb) -> None:
        self.context.close()
        self.playwright_manager.stop()
