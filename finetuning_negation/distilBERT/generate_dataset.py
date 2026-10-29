"""
Generate synthetic dataset for negation parsing fine-tuning.
Uses LLM (nemotron) to parse queries and create labeled data.
"""

import json
import random
from typing import List, Dict
from ollama import chat

# Template variations for negation queries in trade agreements context
QUERY_TEMPLATES = [
    # Basic exclusion
    "{topic} excluding {excluded}",
    "{topic} without {excluded}",
    "{topic} not including {excluded}",
    "{topic} except {excluded}",
    "{topic} except for {excluded}",
    # Negation with action verb
    "{topic} not related to {excluded}",
    "{topic} not covering {excluded}",
    "{topic} not about {excluded}",
    # With country
    "{country} {topic} excluding {excluded}",
    "{topic} in {country} without {excluded}",
    "{country} {topic} not including {excluded}",
    # Multiple exclusions
    "{topic} excluding {excluded} and {excluded2}",
    "{topic} without {excluded} or {excluded2}",
    "{topic} not including {excluded}, {excluded2}, or {excluded3}",
    # Complex patterns
    "information on {topic} but not {excluded}",
    "{topic} provisions excluding {excluded}",
    "regulations on {topic} without {excluded}",
]

# Domain-specific topics from trade agreements
TOPICS = [
    "tariffs",
    "customs procedures",
    "rules of origin",
    "trade remedies",
    "sanitary measures",
    "technical barriers",
    "investment",
    "intellectual property",
    "services",
    "dispute settlement",
    "government procurement",
    "competition policy",
    "e-commerce",
    "labor provisions",
    "environmental provisions",
    "transparency",
    "institutional provisions",
    "general exceptions",
    "security exceptions",
    "final provisions",
]

# Common excluded terms in trade context
EXCLUDED_TERMS = [
    "agriculture",
    "agricultural products",
    "rice",
    "textiles",
    "clothing",
    "apparel",
    "automotive",
    "vehicles",
    "steel",
    "aluminum",
    "services",
    "financial services",
    "banking",
    "investment",
    "dispute settlement",
    "intellectual property",
    "copyright",
    "patents",
    "trademarks",
    "labor provisions",
    "environmental provisions",
    "e-commerce",
    "digital trade",
]

# Countries commonly mentioned in trade agreements
COUNTRIES = [
    "Vietnam",
    "Japan",
    "Korea",
    "China",
    "United States",
    "Australia",
    "New Zealand",
    "Singapore",
    "Thailand",
    "Malaysia",
    "Indonesia",
    "Philippines",
    "ASEAN",
    "EU",
    "India",
    "Canada",
    "Mexico",
    "Chile",
    "Peru",
    "Russia",
]


def generate_random_query() -> str:
    """Generate a random query from templates."""
    template = random.choice(QUERY_TEMPLATES)

    # Prepare template variables
    topic = random.choice(TOPICS)
    excluded = random.choice(EXCLUDED_TERMS)

    # Ensure excluded is not the same as topic
    while excluded.lower() in topic.lower() or topic.lower() in excluded.lower():
        excluded = random.choice(EXCLUDED_TERMS)

    # Build format kwargs
    kwargs = {
        "topic": topic,
        "excluded": excluded,
    }

    # Add country if template needs it
    if "{country}" in template:
        kwargs["country"] = random.choice(COUNTRIES)

    # Add additional excluded terms if template needs them
    if "{excluded2}" in template:
        excluded2 = random.choice(EXCLUDED_TERMS)
        while excluded2 == excluded:
            excluded2 = random.choice(EXCLUDED_TERMS)
        kwargs["excluded2"] = excluded2

    if "{excluded3}" in template:
        excluded3 = random.choice(EXCLUDED_TERMS)
        while excluded3 in [excluded, kwargs.get("excluded2", "")]:
            excluded3 = random.choice(EXCLUDED_TERMS)
        kwargs["excluded3"] = excluded3

    try:
        return template.format(**kwargs)
    except KeyError:
        # Fallback for templates that don't need certain vars
        return template.format(topic=topic, excluded=excluded)


def parse_query_with_llm(query: str, model: str = "nemotron-3-nano:4b") -> Dict:
    """Parse query using LLM to extract positive_query and excluded_terms."""
    prompt = f"""
You are a legal document assistant for a trade agreement search engine.
Analyze the following user query and extract structured information.

Rules:
1. positive_query: The main topic to search for. REMOVE all negation phrases (e.g., "not about", "excluding", "without").
2. excluded_terms: A list of specific concepts the user does NOT want (empty list [] if none).
3. iso_code: A 3-letter ISO country code if a country is mentioned (null if none).

Query: "{query}"

Respond ONLY in valid JSON. No explanation. No markdown.
Example: {{"positive_query": "textile tariffs", "excluded_terms": ["silk", "cotton"], "iso_code": "VNM"}}
"""

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.message.content.strip()

        # Clean markdown if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        return {
            "query": query,
            "positive_query": parsed.get("positive_query", query),
            "excluded_terms": parsed.get("excluded_terms", []),
            "iso_code": parsed.get("iso_code", None),
        }

    except Exception as e:
        print(f"Error parsing query '{query}': {e}")
        return {
            "query": query,
            "positive_query": query,
            "excluded_terms": [],
            "iso_code": None,
        }


def clean_dataset(data: List[Dict]) -> List[Dict]:
    """Clean and filter dataset for quality."""
    cleaned = []
    seen_queries = set()

    for item in data:
        query = item["query"]
        positive = item["positive_query"]
        excluded = item["excluded_terms"]

        # Skip duplicates based on query text
        if query.lower() in seen_queries:
            continue
        seen_queries.add(query.lower())

        # Skip if parsing failed (positive same as query but query has negation words)
        negation_words = [
            "excluding",
            "without",
            "not including",
            "except",
            "except for",
            "not related",
            "not covering",
            "not about",
            "but not",
        ]
        has_negation = any(word in query.lower() for word in negation_words)

        if has_negation and not excluded:
            # LLM failed to extract excluded terms
            print(f"Filtering out (failed parsing): {query}")
            continue

        # Skip if excluded terms still in positive query
        if excluded:
            positive_lower = positive.lower()
            still_has_excluded = any(
                term.lower() in positive_lower for term in excluded
            )
            if still_has_excluded:
                print(f"Filtering out (excluded in positive): {query}")
                continue

        cleaned.append(item)

    return cleaned


def generate_dataset(
    num_samples: int = 1000,
    output_file: str = "synthetic_dataset.json",
    model: str = "nemotron-3-nano:4b",
):
    """Generate synthetic dataset."""
    print(f"Generating {num_samples} samples...")

    dataset = []
    attempts = 0
    max_attempts = num_samples * 3  # Allow for some failed parsings

    while len(dataset) < num_samples and attempts < max_attempts:
        query = generate_random_query()
        print(f"\n[{len(dataset) + 1}/{num_samples}] Parsing: {query}")

        parsed = parse_query_with_llm(query, model)
        print(f"  → Positive: {parsed['positive_query']}")
        print(f"  → Excluded: {parsed['excluded_terms']}")
        print(f"  → ISO Code: {parsed['iso_code']}")

        dataset.append(parsed)
        attempts += 1

    print(f"\n\nCleaning dataset...")
    cleaned_dataset = clean_dataset(dataset)

    print(f"Original: {len(dataset)} samples")
    print(f"Cleaned: {len(cleaned_dataset)} samples")

    # Save to file
    with open(output_file, "w") as f:
        json.dump(cleaned_dataset, f, indent=2)

    print(f"\nSaved to {output_file}")

    return cleaned_dataset


if __name__ == "__main__":
    import sys

    # Allow command line arguments
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output_file = sys.argv[2] if len(sys.argv) > 2 else "synthetic_dataset.json"

    generate_dataset(num_samples=num_samples, output_file=output_file)
