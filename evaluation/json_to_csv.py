import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "rta_id",
    "agreement_title",
    "chapter_title",
    "article_number",
    "article_title",
    "article_text",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return ""


def iter_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "agreements" in payload:
            agreements = payload["agreements"]
        elif "rows" in payload:
            agreements = payload["rows"]
        else:
            agreements = payload.get("data", {}).get("rows", [])
    elif isinstance(payload, list):
        agreements = payload
    else:
        raise ValueError("JSON root must be an object or a list.")

    rows = []
    for agreement in agreements:
        rta_id = get_value(agreement, "rta_id")
        agreement_title = get_value(agreement, "agreement_title", "agreement")

        for article in agreement.get("articles", []):
            rows.append(
                {
                    "rta_id": rta_id,
                    "agreement_title": agreement_title,
                    "chapter_title": get_value(
                        article, "chapter_title", "chapter_name"
                    ),
                    "article_number": get_value(article, "article_number"),
                    "article_title": get_value(article, "article_title", "article_name"),
                    "article_text": get_value(article, "article_text"),
                }
            )

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ground truth candidate JSON into CSV."
    )
    parser.add_argument("input", help="Input JSON file path.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file path. Defaults to the input file name with .csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    payload = load_json(input_path)
    rows = iter_rows(payload)
    write_csv(rows, output_path)

    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
