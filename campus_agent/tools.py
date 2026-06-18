from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import OpenerDirector, Request, build_opener


@dataclass(slots=True)
class CommunitySearchResult:
    title: str
    url: str
    snippet: str
    created_at: str = ""
    updated_at: str = ""
    reply_count: int = 0
    like_count: int = 0
    relevance_score: float = 0.0
    body_loaded: bool = False
    tags: list[str] | None = None
    is_wiki: bool = False
    has_solution: bool = False
    solution_post_number: int = 0
    support_count: int = 0
    primary_object: str = ""
    object_kind: str = ""
    scope: str = ""
    coverage_tags: list[str] | None = None
    redundant_with: str = ""
    ranking_notes: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if self.coverage_tags is None:
            self.coverage_tags = []
        if self.ranking_notes is None:
            self.ranking_notes = {}


@dataclass(slots=True)
class CommunityPost:
    post_number: int
    text: str
    username: str = ""
    created_at: str = ""
    updated_at: str = ""
    is_solution: bool = False


@dataclass(slots=True)
class CommunityDocument:
    title: str
    url: str
    text: str
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] | None = None
    posts: list[CommunityPost] | None = None
    is_wiki: bool = False
    has_solution: bool = False
    solution_post_number: int = 0

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if self.posts is None:
            self.posts = []


class CommunitySearchError(RuntimeError):
    """Raised when the community search backend cannot complete a request."""

    def __init__(self, message: str, *, wait_seconds: int = 0, retryable: bool = False) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds
        self.retryable = retryable


class CommunitySearchTool(Protocol):
    def search(self, query: str, limit: int = 5) -> list[CommunitySearchResult]:
        raise NotImplementedError

    def fetch_topic(self, url: str) -> CommunityDocument | None:
        raise NotImplementedError

    @property
    def last_rate_limit_wait_seconds(self) -> int:
        raise NotImplementedError


class ShuiyuanDiscourseSearchTool:
    """Non-persistent Shuiyuan search via Discourse User-Api-Key."""

    def __init__(
        self,
        *,
        base_url: str = "https://shuiyuan.sjtu.edu.cn",
        user_api_key: str = "",
        user_api_client_id: str = "",
        timeout_seconds: int = 15,
        opener: OpenerDirector | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.user_api_key = user_api_key.strip()
        self.user_api_client_id = user_api_client_id.strip()
        self.timeout_seconds = timeout_seconds
        self._opener = opener or build_opener()
        self.last_rate_limit_wait_seconds = 0
        self.total_rate_limit_wait_seconds = 0
        self._max_auto_wait_seconds = 60
        self._max_total_wait_seconds = 180
        self._max_timeout_retries = 2
        self._progress_callback = progress_callback

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        self._progress_callback = callback

    def search(self, query: str, limit: int = 5) -> list[CommunitySearchResult]:
        term = query.strip()
        if not term:
            return []
        if not self.user_api_key:
            raise CommunitySearchError("missing User-Api-Key for Shuiyuan search")

        payload = self._get_json(
            f"{self.base_url}/search.json?q={urlencode({'q': term})[2:]}",
            headers={
                "Accept": "application/json",
                "User-Api-Key": self.user_api_key,
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                ),
                **(
                    {"User-Api-Client-Id": self.user_api_client_id}
                    if self.user_api_client_id
                    else {}
                ),
            },
        )
        return _parse_discourse_search_results(self.base_url, payload, limit=limit)

    def fetch_topic(self, url: str) -> CommunityDocument | None:
        if not self.user_api_key:
            raise CommunitySearchError("missing User-Api-Key for Shuiyuan search")
        endpoint = _topic_json_endpoint(self.base_url, url)
        if not endpoint:
            return None
        payload = self._get_json(
            endpoint,
            headers={
                "Accept": "application/json",
                "User-Api-Key": self.user_api_key,
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                ),
                **(
                    {"User-Api-Client-Id": self.user_api_client_id}
                    if self.user_api_client_id
                    else {}
                ),
            },
        )
        return _parse_discourse_topic(self.base_url, payload)

    def _get_json(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, object]:
        return self._open_json(Request(url, headers=headers or {}, method="GET"))

    def _open_json(self, request: Request) -> dict[str, object]:
        self.last_rate_limit_wait_seconds = 0
        timeout_attempt = 0
        try:
            while True:
                try:
                    with self._opener.open(request, timeout=self.timeout_seconds) as response:
                        raw = response.read().decode("utf-8")
                    break
                except HTTPError as exc:
                    detail = _read_http_error_detail(exc)
                    wait_seconds = _extract_rate_limit_wait_seconds(detail)
                    if (
                        exc.code == 429
                        and wait_seconds > 0
                        and wait_seconds <= self._max_auto_wait_seconds
                        and self.total_rate_limit_wait_seconds + wait_seconds <= self._max_total_wait_seconds
                    ):
                        self.last_rate_limit_wait_seconds += wait_seconds
                        self.total_rate_limit_wait_seconds += wait_seconds
                        if self._progress_callback is not None:
                            self._progress_callback(
                                f"Shuiyuan 限流，等待 {wait_seconds} 秒后继续（累计等待 {self.total_rate_limit_wait_seconds} 秒）"
                            )
                        time.sleep(wait_seconds)
                        continue
                    raise CommunitySearchError(
                        f"Discourse request failed: HTTP {exc.code} {detail}",
                        wait_seconds=wait_seconds,
                        retryable=exc.code == 429 and wait_seconds > 0,
                    ) from exc
                except (URLError, TimeoutError, OSError, ValueError) as exc:
                    if _is_timeout_error(exc) and timeout_attempt < self._max_timeout_retries:
                        timeout_attempt += 1
                        wait_seconds = min(2 * timeout_attempt, 5)
                        self.last_rate_limit_wait_seconds += wait_seconds
                        self.total_rate_limit_wait_seconds += wait_seconds
                        if self._progress_callback is not None:
                            self._progress_callback(
                                f"Shuiyuan 请求超时，等待 {wait_seconds} 秒后重试（第 {timeout_attempt} 次）"
                            )
                        time.sleep(wait_seconds)
                        continue
                    raise CommunitySearchError(f"Discourse request failed: {exc}") from exc
        except CommunitySearchError:
            raise

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CommunitySearchError(f"Discourse returned non-JSON response: {raw[:200]}") from exc
        if not isinstance(body, dict):
            raise CommunitySearchError("Discourse returned an unexpected response body")
        return body


def build_shuiyuan_search_tool(
    *,
    base_url: str = "https://shuiyuan.sjtu.edu.cn",
    user_api_key: str = "",
    user_api_client_id: str = "",
    timeout_seconds: int = 15,
    progress_callback: Callable[[str], None] | None = None,
) -> CommunitySearchTool:
    return ShuiyuanDiscourseSearchTool(
        base_url=base_url,
        user_api_key=user_api_key,
        user_api_client_id=user_api_client_id,
        timeout_seconds=timeout_seconds,
        progress_callback=progress_callback,
    )


def _parse_discourse_search_results(
    base_url: str,
    payload: dict[str, object],
    *,
    limit: int,
) -> list[CommunitySearchResult]:
    topics = payload.get("topics")
    posts = payload.get("posts")
    topic_items = topics if isinstance(topics, list) else []
    post_items = posts if isinstance(posts, list) else []

    blurbs_by_topic_id: dict[int, str] = {}
    for item in post_items:
        if not isinstance(item, dict):
            continue
        topic_id = item.get("topic_id")
        if not isinstance(topic_id, int):
            continue
        blurb = _normalize_snippet(item.get("blurb") or item.get("excerpt") or "")
        if blurb and topic_id not in blurbs_by_topic_id:
            blurbs_by_topic_id[topic_id] = blurb

    results: list[CommunitySearchResult] = []
    seen_urls: set[str] = set()
    covered_topic_ids: set[int] = set()
    for item in topic_items:
        if not isinstance(item, dict):
            continue
        topic_id = item.get("id")
        title = str(item.get("title") or "").strip()
        if not isinstance(topic_id, int) or not title:
            continue
        slug = str(item.get("slug") or "topic").strip() or "topic"
        url = urljoin(base_url + "/", f"t/{slug}/{topic_id}")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        covered_topic_ids.add(topic_id)
        tags = _normalize_tags(item.get("tags"))
        solution_post_number = _extract_solution_post_number(item)
        results.append(
            CommunitySearchResult(
                title=title,
                url=url,
                snippet=(
                    blurbs_by_topic_id.get(topic_id)
                    or _normalize_snippet(item.get("blurb") or item.get("excerpt") or "")
                    or "未获取到摘要。"
                ),
                created_at=str(item.get("created_at") or ""),
                updated_at=str(item.get("last_posted_at") or item.get("bumped_at") or ""),
                reply_count=_coerce_int(item.get("posts_count")) - 1 if _coerce_int(item.get("posts_count")) > 0 else _coerce_int(item.get("reply_count")),
                like_count=_coerce_int(item.get("like_count")),
                tags=tags,
                is_wiki=_is_wiki_topic(title, tags),
                has_solution=solution_post_number > 0,
                solution_post_number=solution_post_number,
            )
        )
        if len(results) >= limit:
            return results

    for item in post_items:
        if not isinstance(item, dict):
            continue
        topic_id = item.get("topic_id")
        if not isinstance(topic_id, int):
            continue
        if topic_id in covered_topic_ids:
            continue
        slug = str(item.get("topic_slug") or item.get("slug") or "topic").strip() or "topic"
        post_number = _coerce_int(item.get("post_number")) or 1
        title = str(item.get("topic_title") or item.get("title") or f"帖子 {topic_id}").strip()
        url = urljoin(base_url + "/", f"t/{slug}/{topic_id}/{post_number}")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        tags = _normalize_tags(item.get("tags"))
        solution_post_number = _extract_solution_post_number(item)
        results.append(
            CommunitySearchResult(
                title=title,
                url=url,
                snippet=_normalize_snippet(item.get("blurb") or item.get("excerpt") or "") or "未获取到摘要。",
                created_at=str(item.get("created_at") or ""),
                updated_at=str(item.get("updated_at") or ""),
                reply_count=0,
                like_count=_coerce_int(item.get("like_count")),
                tags=tags,
                is_wiki=_is_wiki_topic(title, tags),
                has_solution=solution_post_number > 0,
                solution_post_number=solution_post_number,
            )
        )
        if len(results) >= limit:
            break
    return results


def _topic_json_endpoint(base_url: str, topic_url: str) -> str:
    parsed = urlparse(topic_url)
    match = re.search(r"/t/([^/]+)/(\d+)", parsed.path)
    if not match:
        return ""
    slug, topic_id = match.groups()
    return urljoin(base_url + "/", f"t/{slug}/{topic_id}.json")


def _parse_discourse_topic(
    base_url: str,
    payload: dict[str, object],
) -> CommunityDocument | None:
    topic_id = _coerce_int(payload.get("id"))
    title = str(payload.get("title") or "").strip()
    slug = str(payload.get("slug") or "topic").strip() or "topic"
    posts_root = payload.get("post_stream")
    if not isinstance(posts_root, dict) or not title or topic_id <= 0:
        return None
    post_items = posts_root.get("posts")
    if not isinstance(post_items, list):
        return None

    tags = _normalize_tags(payload.get("tags"))
    solution_post_number = _extract_solution_post_number(payload)
    is_wiki = _is_wiki_topic(title, tags)

    parts: list[str] = []
    parsed_posts: list[CommunityPost] = []
    created_at = ""
    updated_at = ""
    for item in post_items:
        if not isinstance(item, dict):
            continue
        cooked = _normalize_snippet(item.get("cooked") or item.get("blurb") or item.get("excerpt") or "")
        if not cooked:
            continue
        username = str(item.get("username") or "").strip()
        post_number = _coerce_int(item.get("post_number")) or len(parts) + 1
        prefix = f"{username} #{post_number}" if username else f"post #{post_number}"
        parts.append(f"{prefix}: {cooked}")
        is_solution = (
            post_number == solution_post_number
            or _coerce_int(item.get("accepted_answer_post_number")) == post_number
            or _coerce_int(item.get("accepted_solution_post_number")) == post_number
            or bool(item.get("accepted_answer"))
            or bool(item.get("accepted_solution"))
        )
        parsed_posts.append(
            CommunityPost(
                post_number=post_number,
                username=username,
                text=cooked,
                created_at=str(item.get("created_at") or ""),
                updated_at=str(item.get("updated_at") or item.get("created_at") or ""),
                is_solution=is_solution,
            )
        )
        if not created_at:
            created_at = str(item.get("created_at") or "")
        updated_at = str(item.get("updated_at") or item.get("created_at") or updated_at)
    if not parts:
        return None
    return CommunityDocument(
        title=title,
        url=urljoin(base_url + "/", f"t/{slug}/{topic_id}"),
        text="\n\n".join(parts),
        created_at=created_at,
        updated_at=updated_at,
        tags=tags,
        posts=parsed_posts,
        is_wiki=is_wiki,
        has_solution=solution_post_number > 0,
        solution_post_number=solution_post_number,
    )


def _normalize_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(tag).strip() for tag in value if str(tag).strip()]


def _is_wiki_topic(title: str, tags: list[str]) -> bool:
    normalized_title = title.lower()
    normalized_tags = {tag.lower() for tag in tags}
    return "wiki" in normalized_tags or any(term in normalized_title for term in ("[wiki]", "wiki", "指南", "教程", "攻略", "手册"))


def _extract_solution_post_number(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    for key in (
        "accepted_answer_post_number",
        "accepted_solution_post_number",
        "solution_post_number",
        "accepted_post_number",
    ):
        value = _coerce_int(payload.get(key))
        if value > 0:
            return value
    for key in ("accepted_answer", "accepted_solution", "solution"):
        value = payload.get(key)
        if isinstance(value, dict):
            for nested_key in ("post_number", "postNumber"):
                nested_value = _coerce_int(value.get(nested_key))
                if nested_value > 0:
                    return nested_value
    return 0


def _coerce_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _normalize_snippet(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _read_http_error_detail(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def _extract_rate_limit_wait_seconds(detail: str) -> int:
    try:
        body = json.loads(detail)
    except json.JSONDecodeError:
        return 0
    extras = body.get("extras")
    if isinstance(extras, dict):
        wait = extras.get("wait_seconds")
        try:
            return max(0, int(wait))
        except (TypeError, ValueError):
            return 0
    return 0


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text
