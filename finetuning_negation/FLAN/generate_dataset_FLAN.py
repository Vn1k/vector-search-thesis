"""
Generate synthetic dataset for negation parsing - Seq2Seq format (FLAN-T5).
Menggunakan OpenAI API (gpt-4o-mini) dengan JSON Mode.
"""

import json
import random
import os
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# SETUP OPENAI & ENV
# ==========================================
load_dotenv()
client = OpenAI()

# ==========================================
# DATA & TEMPLATES (TIDAK BERUBAH)
# ==========================================
QUERY_TEMPLATES = [
    # ── EXISTING: Basic exclusion ──────────────────────────────────────
    "{topic} excluding {excluded}",
    "{topic} without {excluded}",
    "{topic} not including {excluded}",
    "{topic} except {excluded}",
    "{topic} except for {excluded}",
    "{topic} not related to {excluded}",
    "{topic} not covering {excluded}",
    "{topic} not about {excluded}",
    "{topic} aside from {excluded}",
    "{topic} other than {excluded}",
    "{topic} leaving out {excluded}",
    "{topic} with {excluded} omitted",
    "{topic} — {excluded} not included",
    "everything about {topic} except {excluded}",
    "{topic} where {excluded} is not a factor",
    "{topic} provisions excluding {excluded}",
    "regulations on {topic} without {excluded}",
    "information on {topic} but not {excluded}",

    # ── EXISTING: With country ─────────────────────────────────────────
    "{country} {topic} excluding {excluded}",
    "{topic} in {country} without {excluded}",
    "{country} {topic} not including {excluded}",

    # ── EXISTING: Multiple exclusions ─────────────────────────────────
    "{topic} excluding {excluded} and {excluded2}",
    "{topic} without {excluded} or {excluded2}",
    "{topic} not including {excluded}, {excluded2}, or {excluded3}",

    # ── NEW: Negasi di awal kalimat ────────────────────────────────────
    "exclude {excluded} from {topic}",
    "excluding {excluded}, tell me about {topic}",
    "no {excluded}, just {topic}",
    "not {excluded} — {topic} only",
    "remove {excluded} from {topic} results",
    "filter out {excluded} from {topic}",
    "drop {excluded}, show me {topic}",

    # ── NEW: Kalimat pertanyaan ────────────────────────────────────────
    "what are {topic} that don't include {excluded}?",
    "which {topic} rules exclude {excluded}?",
    "can I get {topic} without the {excluded} part?",
    "what {topic} provisions don't cover {excluded}?",
    "show me {topic} that has nothing to do with {excluded}",
    "where can I find {topic} excluding {excluded}?",
    "is there {topic} information that skips {excluded}?",

    # ── NEW: Conversational / natural language ─────────────────────────
    "I need {topic} but please leave out {excluded}",
    "looking for {topic}, not the part about {excluded}",
    "I want {topic} info, skip {excluded}",
    "give me {topic} and ignore {excluded}",
    "I'm interested in {topic} but not {excluded}",
    "find me {topic} that doesn't mention {excluded}",
    "I only care about {topic}, not {excluded}",
    "search for {topic} and remove any {excluded} content",
    "I need information about {topic}, please omit {excluded}",
    "can you show {topic} and leave {excluded} out?",
    "just {topic}, I don't need anything about {excluded}",
    "focus on {topic} and disregard {excluded}",

    # ── NEW: Implicit negation (tanpa kata negasi eksplisit) ───────────
    "{topic} — {excluded} out of scope",
    "{topic} only, {excluded} not relevant",
    "{topic}, {excluded} not applicable",
    "{topic}; {excluded} is irrelevant here",
    "{topic} — ignore {excluded}",
    "{topic}, skip {excluded}",
    "{topic} minus {excluded}",
    "{topic} less {excluded}",

    # ── NEW: Multiple exclusions — berbeda pola ────────────────────────
    "{topic}, but not {excluded}, not {excluded2}",
    "{topic} — no {excluded}, no {excluded2}",
    "{topic} without {excluded} and without {excluded2}",
    "{topic} except {excluded} and except {excluded2}",
    "{topic}, omit {excluded} and {excluded2}",
    "I want {topic} but exclude both {excluded} and {excluded2}",

    # ── NEW: Informal / disingkat ──────────────────────────────────────
    "{topic} excl. {excluded}",
    "{topic} w/o {excluded}",
    "{topic} - no {excluded}",
    "{topic} (not {excluded})",
    "{topic} [excluding {excluded}]",
    "{topic} ^{excluded}",           # gaya boolean search

    # ── NEW: Country + conversational ─────────────────────────────────
    "in {country}, {topic} without {excluded}",
    "{country} {topic} — skip {excluded}",
    "for {country}: {topic} excluding {excluded}",
    "{topic} for {country} but not {excluded}",
    "I need {country} {topic} info, leave out {excluded}",

    # ── NEW: Scope framing ─────────────────────────────────────────────
    "{topic} within scope, {excluded} out of scope",
    "{topic} in scope excluding {excluded}",
    "{topic} coverage without {excluded}",
    "{topic} provisions, {excluded} exempted",
    "{topic} articles not touching {excluded}",
    "{topic} chapters that avoid {excluded}",
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

PROVISION_TREE = {
    "Trade in Goods": [
        "General Issues Under Trade in Goods",
        "Agriculture",
        "Anti-Dumping",
        "Quantitative Restrictions on Exports and Imports",
        "Rules of Origin",
        "Sanitary and Phytosanitary Measures",
        "Subsidies",
        "Technical Barriers to Trade",
        "Trade Facilitation",
        "Trade Related Investment",
    ],
    "Digital Trade / Electronic Commerce": [
        "General Issues Under Digital Trade",
        "Paperless Trade",
        "Data Related Digital Trade Policy",
    ],
    "Trade in Services": [
        "General Issues Under Trade in Services",
    ],
    "Competition Policy": [],
    "Development": [],
    "Energy": [],
    "Environment": [],
    "Gender": [],
    "Government Procurement": [],
    "Intellectual Property Rights": [],
    "Labour": [],
    "Micro, Small and Medium Sized Enterprises": [],
    "State-Owned Enterprise": [],
    "Dispute Settlement": [],
}

EXCLUDED_TERMS = [
    # Komoditas
    "rice", "wheat", "steel", "aluminum", "textiles", "clothing", "apparel",
    "automotive", "vehicles", "agricultural products", "fisheries products",
    "pharmaceutical products", "medical devices",
    # Konsep legal trade
    "MFN treatment", "national treatment", "market access",
    "non-tariff barriers", "anti-circumvention", "safeguard measures",
    "countervailing measures", "anti-dumping duties",
    # Bagian dokumen
    "definitions", "annexes", "schedules", "transitional provisions",
    "sunset clauses", "rules of procedure", "general exceptions",
    "security exceptions", "final provisions",
    # Sektor
    "financial services", "banking", "insurance", "telecommunications",
    "maritime transport", "air transport", "professional services",
    "digital services", "e-commerce platforms",
    # Lainnya
    "investment arbitration", "ISDS", "MFN clause",
    "transparency requirements", "notification obligations",
]

COUNTRIES = [
    "Vietnam", "Japan", "Korea", "China", "United States", "Australia",
    "New Zealand", "Singapore", "Thailand", "Malaysia", "Indonesia",
    "Philippines", "ASEAN", "EU", "India", "Canada", "Mexico",
]

ALL_PARENTS = list(PROVISION_TREE.keys())
ALL_CHILDREN = [child for children in PROVISION_TREE.values() for child in children]
ALL_PROVISIONS = ALL_PARENTS + ALL_CHILDREN

# ==========================================
# FUNGSI-FUNGSI UTAMA
# ==========================================

def pick_topic() -> str:
    """
    Pilih topic secara random dengan distribusi yang realistis:
    - 30% parent provision saja        → "Trade in Goods"
    - 40% sub-provision saja           → "Rules of Origin"
    - 30% parent + sub (konteks penuh) → "Trade in Goods — Rules of Origin"
    """
    roll = random.random()

    if roll < 0.30:
        # Parent saja
        return random.choice(ALL_PARENTS)

    elif roll < 0.70:
        # Sub-provision saja (hanya parent yang punya children)
        parents_with_children = [p for p, c in PROVISION_TREE.items() if c]
        parent = random.choice(parents_with_children)
        return random.choice(PROVISION_TREE[parent])

    else:
        # Parent + sub-provision digabung
        parents_with_children = [p for p, c in PROVISION_TREE.items() if c]
        parent = random.choice(parents_with_children)
        child  = random.choice(PROVISION_TREE[parent])
        # Variasi format penggabungan supaya model belajar berbagai cara penulisan
        fmt = random.choice([
            f"{parent} — {child}",
            f"{child} under {parent}",
            f"{parent} specifically {child}",
            f"{child} provisions in {parent}",
        ])
        return fmt


def pick_excluded(topic: str) -> str:
    """
    Pilih excluded term yang tidak overlap dengan topic.
    Excluded bisa dari provision lain atau dari EXCLUDED_TERMS domain-specific.
    """
    # Gabungkan semua provision sebagai kandidat excluded
    candidates = ALL_PROVISIONS + EXCLUDED_TERMS

    excluded = random.choice(candidates)
    attempts = 0
    while (
        excluded.lower() in topic.lower()
        or topic.lower() in excluded.lower()
        and attempts < 20
    ):
        excluded = random.choice(candidates)
        attempts += 1

    return excluded

def generate_random_query() -> str:
    template = random.choice(QUERY_TEMPLATES)

    topic    = pick_topic()
    excluded = pick_excluded(topic)

    kwargs = {"topic": topic, "excluded": excluded}

    if "{country}" in template:
        kwargs["country"] = random.choice(COUNTRIES)

    if "{excluded2}" in template:
        excluded2 = pick_excluded(topic)
        while excluded2 == excluded:
            excluded2 = pick_excluded(topic)
        kwargs["excluded2"] = excluded2

    if "{excluded3}" in template:
        excluded3 = pick_excluded(topic)
        while excluded3 in [excluded, kwargs.get("excluded2", "")]:
            excluded3 = pick_excluded(topic)
        kwargs["excluded3"] = excluded3

    try:
        return template.format(**kwargs)
    except KeyError:
        return template.format(topic=topic, excluded=excluded)


def parse_query_with_llm(query: str, model: str = "gpt-4o-mini") -> Dict:
    prompt = f"""You are a structured data extraction engine for a trade agreement search system.

Your task: parse a user query into 3 fields.

=== OUTPUT SCHEMA ===
{{
  "positive_query": "<main topic to search for, with ALL negation phrases and their objects fully removed>",
  "excluded_terms": ["<term1>", "<term2>"],  // empty list [] if none
  "iso_code": "<ISO 3166-1 alpha-3 code or null>"
}}

=== STRICT RULES ===
1. POSITIVE QUERY: Keep ONLY the core topic. Remove the negation phrase AND the excluded term(s) from the positive_query.
   - WRONG: "trade remedies covering patents"   ← excluded term leaked in
   - WRONG: "labor provisions including agricultural products"  ← negated term leaked in
   - CORRECT: just "trade remedies" / "labor provisions"

2. NEGATION PHRASES TO DETECT (all of these signal exclusion):
   "excluding", "without", "not including", "except", "except for",
   "not related to", "not covering", "not about", "but not",
   "aside from", "other than", "leaving out", "omitted",
   "not included", "— X not included", "where X is not a factor"

3. "— X not included" pattern: the part after "—" and before "not included" is the excluded term.
   Example: "sanitary measures — digital trade not included"
   → positive_query: "sanitary measures", excluded_terms: ["digital trade"]

4. "where X is not a factor" pattern: X is excluded.
   Example: "rules of origin where agriculture is not a factor"
   → positive_query: "rules of origin", excluded_terms: ["agriculture"]

5. "not about / not covering / not including" pattern: the object after the phrase is excluded.
   Example: "institutional provisions not about agricultural products"
   → positive_query: "institutional provisions", excluded_terms: ["agricultural products"]
   Example: "Vietnam sanitary measures not including clothing"
   → positive_query: "Vietnam sanitary measures", excluded_terms: ["clothing"]

6. ISO CODE: Only standard country names map to ISO codes (e.g. Vietnam→VNM, Japan→JPN).
   Blocs like "ASEAN", "EU" are NOT countries → iso_code: null

=== FEW-SHOT EXAMPLES ===
Query: "government procurement — vehicles not included"
Output: {{"positive_query": "government procurement", "excluded_terms": ["vehicles"], "iso_code": null}}

Query: "trade remedies not covering patents"
Output: {{"positive_query": "trade remedies", "excluded_terms": ["patents"], "iso_code": null}}

Query: "final provisions aside from clothing"
Output: {{"positive_query": "final provisions", "excluded_terms": ["clothing"], "iso_code": null}}

Query: "rules of origin where agriculture is not a factor"
Output: {{"positive_query": "rules of origin", "excluded_terms": ["agriculture"], "iso_code": null}}

Query: "institutional provisions not about agricultural products"
Output: {{"positive_query": "institutional provisions", "excluded_terms": ["agricultural products"], "iso_code": null}}

Query: "Vietnam sanitary measures not including clothing"
Output: {{"positive_query": "Vietnam sanitary measures", "excluded_terms": ["clothing"], "iso_code": "VNM"}}

Query: "ASEAN trade remedies not including agricultural products"
Output: {{"positive_query": "ASEAN trade remedies", "excluded_terms": ["agricultural products"], "iso_code": null}}

Query: "services not including steel, safeguards, or textiles"
Output: {{"positive_query": "services", "excluded_terms": ["steel", "safeguards", "textiles"], "iso_code": null}}

Query: "competition policy in Mexico without patents"
Output: {{"positive_query": "competition policy in Mexico", "excluded_terms": ["patents"], "iso_code": "MEX"}}

=== NOW PARSE THIS ===
Query: "{query}"
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extraction engine. "
                        "Follow all rules exactly. "
                        "Respond ONLY with a valid JSON object. No explanation, no markdown."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

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


# ==========================================
# DIUBAH: Format output pipe-delimited, bukan JSON
# Alasan: T5 tokenizer (SentencePiece) tidak punya { } [ ] dalam vocab-nya,
#         sehingga model gagal generate JSON secara konsisten.
#         Format "positive: X | excluded: Y | country: Z" jauh lebih mudah
#         di-generate oleh T5 karena semua karakternya familiar.
# ==========================================
def build_seq2seq_example(item: Dict) -> Dict:
    """Mengubah format untuk FLAN-T5 — pipe-delimited bukan JSON."""
    excluded_str = ", ".join(item["excluded_terms"]) if item["excluded_terms"] else "none"
    iso_str = item["iso_code"] if item["iso_code"] else "none"

    input_text = (
        "Parse the trade agreement search query. "
        "Extract the positive topic and any excluded terms. "
        f"Query: {item['query']}"
    )

    # Format: "positive: <X> | excluded: <Y> | country: <Z>"
    # Mudah di-parse kembali dan mudah di-generate oleh T5
    output_text = (
        f"positive: {item['positive_query']} | "
        f"excluded: {excluded_str} | "
        f"country: {iso_str}"
    )

    return {
        "input": input_text,
        "output": output_text,
        "meta": item,
    }


def clean_dataset(data: List[Dict]) -> List[Dict]:
    """TIDAK BERUBAH: Membersihkan data yang logic-nya gagal."""
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
    num_samples: int = 400,
    output_file: str = "synthetic_dataset_3.json",
    model: str = "gpt-4o-mini",
):
    print(f"Generating {num_samples} samples using {model}...")

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

    seq2seq_data = [build_seq2seq_example(item) for item in cleaned]

    print(f"Original  : {len(dataset)} samples")
    print(f"Cleaned   : {len(cleaned)} samples")
    print(f"Seq2seq   : {len(seq2seq_data)} samples")

    with open(output_file, "w") as f:
        json.dump(seq2seq_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_file}")
    return seq2seq_data


if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    output_file = sys.argv[2] if len(sys.argv) > 2 else "synthetic_dataset.json"
    generate_dataset(num_samples=num_samples, output_file=output_file)