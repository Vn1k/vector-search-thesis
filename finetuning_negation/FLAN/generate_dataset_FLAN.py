"""
Generate synthetic dataset for negation parsing - Seq2Seq format (FLAN-T5).
"""

import json
import random
from typing import List, Dict
from ollama import chat

QUERY_TEMPLATES = [
    "{topic} excluding {excluded}",
    "{topic} without {excluded}",
    "{topic} not including {excluded}",
    "{topic} except {excluded}",
    "{topic} except for {excluded}",
    "{topic} not related to {excluded}",
    "{topic} not covering {excluded}",
    "{topic} not about {excluded}",
    "{country} {topic} excluding {excluded}",
    "{topic} in {country} without {excluded}",
    "{country} {topic} not including {excluded}",
    "{topic} excluding {excluded} and {excluded2}",
    "{topic} without {excluded} or {excluded2}",
    "{topic} not including {excluded}, {excluded2}, or {excluded3}",
    "information on {topic} but not {excluded}",
    "{topic} provisions excluding {excluded}",
    "regulations on {topic} without {excluded}",
    # Tambah pola lebih natural/ambigu supaya model lebih flexible
    "{topic} aside from {excluded}",
    "{topic} other than {excluded}",
    "{topic} leaving out {excluded}",
    "{topic} with {excluded} omitted",
    "{topic} — {excluded} not included",
    "everything about {topic} except {excluded}",
    "{topic} where {excluded} is not a factor",
]

TOPICS = [
    "tariffs", "customs procedures", "rules of origin", "trade remedies",
    "sanitary measures", "technical barriers", "investment",
    "intellectual property", "services", "dispute settlement",
    "government procurement", "competition policy", "e-commerce",
    "labor provisions", "environmental provisions", "transparency",
    "institutional provisions", "general exceptions", "security exceptions",
    "final provisions",
]

EXCLUDED_TERMS = [
    "agriculture", "agricultural products", "rice", "textiles", "clothing",
    "apparel", "automotive", "vehicles", "steel", "aluminum", "services",
    "financial services", "banking", "investment", "dispute settlement",
    "intellectual property", "copyright", "patents", "trademarks",
    "labor provisions", "environmental provisions", "e-commerce",
    "digital trade", "anti-dumping", "countervailing measures", "safeguards",
]

COUNTRIES = [
    "Vietnam", "Japan", "Korea", "China", "United States", "Australia",
    "New Zealand", "Singapore", "Thailand", "Malaysia", "Indonesia",
    "Philippines", "ASEAN", "EU", "India", "Canada", "Mexico",
]


def generate_random_query() -> str:
    template = random.choice(QUERY_TEMPLATES)
    topic = random.choice(TOPICS)
    excluded = random.choice(EXCLUDED_TERMS)

    while excluded.lower() in topic.lower() or topic.lower() in excluded.lower():
        excluded = random.choice(EXCLUDED_TERMS)

    kwargs = {"topic": topic, "excluded": excluded}

    if "{country}" in template:
        kwargs["country"] = random.choice(COUNTRIES)

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
        return template.format(topic=topic, excluded=excluded)


def parse_query_with_llm(query: str, model: str = "nemotron-3-nano:4b") -> Dict:
    prompt = f"""
You are a legal document assistant for a trade agreement search engine.
Analyze the following user query and extract structured information.

Rules:
1. positive_query: The main topic to search for. REMOVE all negation phrases.
2. excluded_terms: A list of specific concepts the user does NOT want (empty list [] if none).
   IMPORTANT: Keep multi-word phrases together (e.g., "anti-dumping" stays as one term, not split).
3. iso_code: A 3-letter ISO country code if a country is mentioned (null if none).

Query: "{query}"

Respond ONLY in valid JSON. No explanation. No markdown.
Example: {{"positive_query": "textile tariffs", "excluded_terms": ["anti-dumping", "cotton"], "iso_code": "VNM"}}
"""
    try:
        response = chat(model=model, messages=[{"role": "user", "content": prompt}])
        raw = response.message.content.strip()

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


def build_seq2seq_example(item: Dict) -> Dict:
    """
    Convert parsed item ke format input/output untuk FLAN-T5.
    Input  : natural language instruction + query
    Output : JSON string
    """
    input_text = (
        "Parse the trade agreement search query. "
        "Extract the positive topic and any excluded terms. "
        f"Query: {item['query']}"
    )

    output_json = {
        "positive_query": item["positive_query"],
        "excluded_terms": item["excluded_terms"],
        "iso_code": item["iso_code"],
    }

    return {
        "input": input_text,
        "output": json.dumps(output_json, ensure_ascii=False),
        # Simpan metadata asli untuk debugging
        "meta": item,
    }


def clean_dataset(data: List[Dict]) -> List[Dict]:
    cleaned = []
    seen_queries = set()

    negation_words = [
        "excluding", "without", "not including", "except", "except for",
        "not related", "not covering", "not about", "but not",
        "aside from", "other than", "leaving out", "omitted",
    ]

    for item in data:
        query = item["query"]
        positive = item["positive_query"]
        excluded = item["excluded_terms"]

        if query.lower() in seen_queries:
            continue
        seen_queries.add(query.lower())

        has_negation = any(word in query.lower() for word in negation_words)

        if has_negation and not excluded:
            print(f"Filtering out (failed parsing): {query}")
            continue

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
    print(f"Generating {num_samples} samples...")

    dataset = []
    attempts = 0
    max_attempts = num_samples * 3

    while len(dataset) < num_samples and attempts < max_attempts:
        query = generate_random_query()
        print(f"\n[{len(dataset) + 1}/{num_samples}] Parsing: {query}")

        parsed = parse_query_with_llm(query, model)
        print(f"  → Positive : {parsed['positive_query']}")
        print(f"  → Excluded : {parsed['excluded_terms']}")
        print(f"  → ISO Code : {parsed['iso_code']}")

        dataset.append(parsed)
        attempts += 1

    print("\nCleaning dataset...")
    cleaned = clean_dataset(dataset)

    # Convert ke format seq2seq
    seq2seq_data = [build_seq2seq_example(item) for item in cleaned]

    print(f"Original  : {len(dataset)} samples")
    print(f"Cleaned   : {len(cleaned)} samples")
    print(f"Seq2seq   : {len(seq2seq_data)} samples")

    with open(output_file, "w") as f:
        json.dump(seq2seq_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_file}")
    return seq2seq_data


if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output_file = sys.argv[2] if len(sys.argv) > 2 else "synthetic_dataset.json"
    generate_dataset(num_samples=num_samples, output_file=output_file)