"""tools/news_tools.py – Real-time news headlines via NewsAPI."""
from __future__ import annotations

import logging

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# NewsAPI free plan does not reliably return results for country=in.
# For India-specific categories we use the /everything endpoint with keyword
# search (requires q=, language=, sortBy=) which works on the free plan.
# World/general uses /top-headlines with language=en (no country filter).
_CATEGORY_CONFIG: dict[str, dict] = {
    "india":         {"endpoint": "everything", "q": "india news today",               "language": "en", "sortBy": "publishedAt"},
    "sports":        {"endpoint": "everything", "q": "cricket india sports news",      "language": "en", "sortBy": "publishedAt"},
    "world":         {"endpoint": "top-headlines",                                     "language": "en"},
    "business":      {"endpoint": "everything", "q": "india business economy market",  "language": "en", "sortBy": "publishedAt"},
    "tech":          {"endpoint": "everything", "q": "india technology startup AI",    "language": "en", "sortBy": "publishedAt"},
    "entertainment": {"endpoint": "everything", "q": "india bollywood entertainment",  "language": "en", "sortBy": "publishedAt"},
    "health":        {"endpoint": "everything", "q": "india health medical news",      "language": "en", "sortBy": "publishedAt"},
    "science":       {"endpoint": "everything", "q": "india science research space",   "language": "en", "sortBy": "publishedAt"},
}


async def fetch_news(
    category: str = "india",
    page_size: int = 8,
) -> dict:
    """
    Fetch top headlines from NewsAPI.

    Uses /everything with keyword search for India-specific categories,
    and /top-headlines for the World category.
    """
    if not settings.news_api_key:
        return {
            "category": category,
            "articles": [],
            "error": "NewsAPI key not configured. Add NEWS_API_KEY to Railway variables.",
        }

    cfg = _CATEGORY_CONFIG.get(category.lower(), _CATEGORY_CONFIG["india"])
    endpoint = cfg.get("endpoint", "top-headlines")

    params: dict = {
        "apiKey": settings.news_api_key,
        "pageSize": min(page_size, 10),
    }

    if endpoint == "everything":
        params["q"] = cfg["q"]
        params["language"] = cfg.get("language", "en")
        params["sortBy"] = cfg.get("sortBy", "publishedAt")
    else:
        # top-headlines (World)
        if "language" in cfg:
            params["language"] = cfg["language"]

    url = f"{settings.news_api_base_url}/{endpoint}"
    logger.info("NewsAPI request: %s %s", endpoint, {k: v for k, v in params.items() if k != 'apiKey'})

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        raw_count = len(data.get("articles", []))
        articles = [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "description": a.get("description") or "",
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in data.get("articles", [])
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]

        logger.info("NewsAPI response: category=%s raw=%d filtered=%d", category, raw_count, len(articles))
        return {"category": category, "articles": articles, "total": len(articles)}

    except Exception as exc:
        logger.error("NewsAPI error for category=%s: %s", category, exc)
        return {"category": category, "articles": [], "error": str(exc)}
