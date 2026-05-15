"""tools/news_tools.py – Real-time news headlines via NewsAPI."""
from __future__ import annotations

import httpx

from config import get_settings

settings = get_settings()

# Maps UI/agent category names → NewsAPI params
_CATEGORY_CONFIG: dict[str, dict] = {
    "india":         {"country": "in", "category": "general"},
    "sports":        {"country": "in", "category": "sports"},
    "world":         {"country": "",   "category": "general"},
    "business":      {"country": "in", "category": "business"},
    "tech":          {"country": "in", "category": "technology"},
    "entertainment": {"country": "in", "category": "entertainment"},
    "health":        {"country": "in", "category": "health"},
    "science":       {"country": "in", "category": "science"},
}


async def fetch_news(
    category: str = "india",
    query: str = "",
    page_size: int = 8,
) -> dict:
    """
    Fetch top headlines from NewsAPI.

    Args:
        category: One of india, sports, world, business, tech,
                  entertainment, health, science.
        query:    Optional keyword to narrow results.
        page_size: Number of articles to return (max 10).

    Returns:
        dict with 'category', 'articles' list, and 'total'.
    """
    if not settings.news_api_key:
        return {
            "category": category,
            "articles": [],
            "error": "NewsAPI key not configured. Add NEWS_API_KEY to Railway variables.",
        }

    cfg = _CATEGORY_CONFIG.get(category.lower(), _CATEGORY_CONFIG["india"])
    params: dict = {
        "apiKey": settings.news_api_key,
        "pageSize": min(page_size, 10),
        "language": "en",
        "category": cfg["category"],
    }
    if cfg["country"]:
        params["country"] = cfg["country"]
    if query:
        params["q"] = query

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                settings.news_api_base_url + "/top-headlines",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

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

        return {"category": category, "articles": articles, "total": len(articles)}

    except Exception as exc:
        return {"category": category, "articles": [], "error": str(exc)}
