import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000/api/v1/search"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def fetch_from_api(
    query: str,
    api_url: str,
    agreement_limit: int,
    threshold: float,
) -> list[dict[str, Any]]:
    response = requests.post(
        api_url,
        json={
            "query": query,
            "limit": agreement_limit,
            "threshold": threshold,
        },
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    return payload.get("data", {}).get("rows", [])


def fetch_from_search_function(
    query: str,
    agreement_limit: int,
    threshold: float,
) -> list[dict[str, Any]]:
    from src.search_engine import get_search_engine

    return get_search_engine().search(
        query=query,
        limit=agreement_limit,
        threshold=threshold,
    )


def to_ground_truth_candidates(
    query: str,
    rows: list[dict[str, Any]],
    top_articles: int,
) -> dict[str, Any]:
    agreements = []

    for row in rows:
        articles = []
        for article in row.get("articles", [])[:top_articles]:
            articles.append(
                {
                    "chapter_name": article.get("chapter_name"),
                    "article_id": article.get("article_id"),
                    "article_number": article.get("article_number"),
                    "article_name": article.get("article_name"),
                    "article_text": article.get("article_text"),
                    "confidence": article.get("confidence"),
                }
            )

        agreements.append(
            {
                "rta_id": row.get("rta_id"),
                "agreement": row.get("agreement"),
                "articles": articles,
            }
        )

    return {
        "query": query,
        "agreements": agreements,
    }


def write_json(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Saved {output_path}")
    else:
        print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export top search articles per agreement for ground truth collection."
    )
    parser.add_argument("query", help="Search query.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--agreement-limit",
        type=int,
        default=10,
        help="Number of agreements to export.",
    )
    parser.add_argument(
        "--top-articles",
        type=int,
        default=3,
        help="Number of top articles to keep for each agreement.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum search score threshold.",
    )
    parser.add_argument(
        "--source",
        choices=["api", "function"],
        default="api",
        help="Use running FastAPI endpoint or call the search function directly.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Search endpoint URL when --source api is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "api":
        rows = fetch_from_api(
            query=args.query,
            api_url=args.api_url,
            agreement_limit=args.agreement_limit,
            threshold=args.threshold,
        )
    else:
        rows = fetch_from_search_function(
            query=args.query,
            agreement_limit=args.agreement_limit,
            threshold=args.threshold,
        )

    payload = to_ground_truth_candidates(
        query=args.query,
        rows=rows,
        top_articles=args.top_articles,
    )
    write_json(payload, args.output)


if __name__ == "__main__":
    main()
