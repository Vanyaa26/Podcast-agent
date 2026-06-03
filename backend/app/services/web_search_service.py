"""Web search for the interrupt agent.

Used when the paper KB does not define a term the listener asks about (e.g.
"What is RAG?" on a modular-RAG paper). Results are labeled as external web
background, not paper evidence.

Backends (first match wins):
  1. Tavily — if TAVILY_API_KEY is set (best quality for agents)
  2. DuckDuckGo — no extra key; good enough for demo definitions
"""
from __future__ import annotations

import logging
import re
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.config import TAVILY_API_KEY, WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)

_SNIPPET_MAX = 320


def _trim(text: str, limit: int = _SNIPPET_MAX) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


class WebSearchService:
    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        limit = max_results or WEB_SEARCH_MAX_RESULTS
        limit = max(1, min(int(limit), 5))

        if TAVILY_API_KEY:
            try:
                hits = self._search_tavily(query, limit)
                if hits:
                    return hits
            except Exception as exc:
                logger.warning("Tavily search failed, falling back: %s", exc)

        try:
            return self._search_duckduckgo(query, limit)
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

    def _search_tavily(self, query: str, limit: int) -> list[dict[str, Any]]:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        hits: list[dict[str, Any]] = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url or "Web result").strip()
            snippet = _trim(str(item.get("content") or ""))
            if not snippet and not title:
                continue
            hits.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "source": "web",
                }
            )
        return hits[:limit]

    def _search_duckduckgo(self, query: str, limit: int) -> list[dict[str, Any]]:
        """DuckDuckGo HTML results — no API key required."""
        response = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "b": "", "kl": "wt-wt", "df": ""},
            headers={"User-Agent": "agentic-podcast-rag/0.1"},
            timeout=20.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        html = response.text

        hits: list[dict[str, Any]] = []
        # Each result block: <a class="result__a" href="...">title</a> ... <a class="result__snippet">...</a>
        for block in re.finditer(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</',
            html,
            re.DOTALL | re.IGNORECASE,
        ):
            url = self._normalize_ddg_url(block.group(1))
            title = _trim(unescape(re.sub(r"<[^>]+>", "", block.group(2))), 120)
            snippet = _trim(unescape(re.sub(r"<[^>]+>", "", block.group(3))))
            if not title and not snippet:
                continue
            hits.append({"title": title or "Web result", "snippet": snippet, "url": url, "source": "web"})
            if len(hits) >= limit:
                break

        if hits:
            return hits

        # Instant-answer fallback for definition-style queries.
        ia = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=15.0,
        )
        ia.raise_for_status()
        data = ia.json()
        abstract = _trim(str(data.get("AbstractText") or ""))
        if abstract:
            return [
                {
                    "title": str(data.get("Heading") or query),
                    "snippet": abstract,
                    "url": str(data.get("AbstractURL") or ""),
                    "source": "web",
                }
            ]
        return []

    @staticmethod
    def _normalize_ddg_url(href: str) -> str:
        href = unescape(href or "")
        if "uddg=" in href:
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if qs.get("uddg"):
                return unquote(qs["uddg"][0])
        return href
