"""
Generate synthetic dataset untuk DistilBERT negation parsing (BIO Tagging).
LLM yang digunakan sama dengan versi FLAN-T5 (gpt-4o-mini) untuk perbandingan yang fair.

Format output: BIO token classification
- B-POS : Beginning of positive query
- I-POS : Inside positive query
- B-NEG : Beginning of excluded term
- I-NEG : Inside excluded term
- O     : Outside (kata negasi, connector, dll)
"""

import json
import random
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# SETUP
# ==========================================
load_dotenv()
client = OpenAI()

# ==========================================
# TEMPLATES & DATA — IDENTIK dengan versi FLAN-T5
# Sengaja tidak diubah agar dataset comparable
# ==========================================
QUERY_TEMPLATES = [
    # ── Basic exclusion ────────────────────────────────────────────────
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
    # ── With country ───────────────────────────────────────────────────
    "{country} {topic} excluding {excluded}",
    "{topic} in {country} without {excluded}",
    "{country} {topic} not including {excluded}",
    # ── Multiple exclusions ────────────────────────────────────────────
    "{topic} excluding {excluded} and {excluded2}",
    "{topic} without {excluded} or {excluded2}",
    "{topic} not including {excluded}, {excluded2}, or {excluded3}",
    # ── Negasi di awal ─────────────────────────────────────────────────
    "exclude {excluded} from {topic}",
    "excluding {excluded}, tell me about {topic}",
    "no {excluded}, just {topic}",
    "not {excluded} — {topic} only",
    "remove {excluded} from {topic} results",
    "filter out {excluded} from {topic}",
    "drop {excluded}, show me {topic}",
    # ── Kalimat pertanyaan ─────────────────────────────────────────────
    "what are {topic} that don't include {excluded}?",
    "which {topic} rules exclude {excluded}?",
    "can I get {topic} without the {excluded} part?",
    "what {topic} provisions don't cover {excluded}?",
    "show me {topic} that has nothing to do with {excluded}",
    "where can I find {topic} excluding {excluded}?",
    "is there {topic} information that skips {excluded}?",
    # ── Conversational ─────────────────────────────────────────────────
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
    # ── Implicit negation ──────────────────────────────────────────────
    "{topic} — {excluded} out of scope",
    "{topic} only, {excluded} not relevant",
    "{topic}, {excluded} not applicable",
    "{topic}; {excluded} is irrelevant here",
    "{topic} — ignore {excluded}",
    "{topic}, skip {excluded}",
    "{topic} minus {excluded}",
    "{topic} less {excluded}",
    # ── Multiple exclusions berbeda pola ───────────────────────────────
    "{topic}, but not {excluded}, not {excluded2}",
    "{topic} — no {excluded}, no {excluded2}",
    "{topic} without {excluded} and without {excluded2}",
    "{topic} except {excluded} and except {excluded2}",
    "{topic}, omit {excluded} and {excluded2}",
    "I want {topic} but exclude both {excluded} and {excluded2}",
    # ── Informal ───────────────────────────────────────────────────────
    "{topic} excl. {excluded}",
    "{topic} w/o {excluded}",
    "{topic} - no {excluded}",
    "{topic} (not {excluded})",
    "{topic} [excluding {excluded}]",
    "{topic} ^{excluded}",
    # ── Country + conversational ───────────────────────────────────────
    "in {country}, {topic} without {excluded}",
    "{country} {topic} — skip {excluded}",
    "for {country}: {topic} excluding {excluded}",
    "{topic} for {country} but not {excluded}",
    "I need {country} {topic} info, leave out {excluded}",
    # ── Scope framing ──────────────────────────────────────────────────
    "{topic} within scope, {excluded} out of scope",
    "{topic} in scope excluding {excluded}",
    "{topic} coverage without {excluded}",
    "{topic} provisions, {excluded} exempted",
    "{topic} articles not touching {excluded}",
    "{topic} chapters that avoid {excluded}",
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
    "rice", "wheat", "steel", "aluminum", "textiles", "clothing", "apparel",
    "automotive", "vehicles", "agricultural products", "fisheries products",
    "pharmaceutical products", "medical devices",
    "MFN treatment", "national treatment", "market access",
    "non-tariff barriers", "anti-circumvention", "safeguard measures",
    "countervailing measures", "anti-dumping duties",
    "definitions", "annexes", "schedules", "transitional provisions",
    "sunset clauses", "rules of procedure", "general exceptions",
    "security exceptions", "final provisions",
    "financial services", "banking", "insurance", "telecommunications",
    "maritime transport", "air transport", "professional services",
    "digital services", "e-commerce platforms",
    "investment arbitration", "ISDS", "MFN clause",
    "transparency requirements", "notification obligations",
]

COUNTRIES = [
    "Vietnam", "Japan", "Korea", "China", "United States", "Australia",
    "New Zealand", "Singapore", "Thailand", "Malaysia", "Indonesia",
    "Philippines", "ASEAN", "EU", "India", "Canada", "Mexico",
]

ALL_PARENTS    = list(PROVISION_TREE.keys())
ALL_CHILDREN   = [c for children in PROVISION_TREE.values() for c in children]
ALL_PROVISIONS = ALL_PARENTS + ALL_CHILDREN


# ==========================================
# TOPIC & EXCLUDED PICKER — IDENTIK dengan FLAN-T5
# ==========================================

def pick_topic() -> str:
    roll = random.random()
    if roll < 0.30:
        return random.choice(ALL_PARENTS)
    elif roll < 0.70:
        parents_with_children = [p for p, c in PROVISION_TREE.items() if c]
        parent = random.choice(parents_with_children)
        return random.choice(PROVISION_TREE[parent])
    else:
        parents_with_children = [p for p, c in PROVISION_TREE.items() if c]
        parent = random.choice(parents_with_children)
        child  = random.choice(PROVISION_TREE[parent])
        fmt = random.choice([
            f"{parent} — {child}",
            f"{child} under {parent}",
            f"{parent} specifically {child}",
            f"{child} provisions in {parent}",
        ])
        return fmt


def pick_excluded(topic: str) -> str:
    candidates = ALL_PROVISIONS + EXCLUDED_TERMS
    excluded   = random.choice(candidates)
    attempts   = 0
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
    kwargs   = {"topic": topic, "excluded": excluded}

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


# ==========================================
# LLM PARSER — IDENTIK dengan versi FLAN-T5
# Sengaja tidak diubah untuk fair comparison
# ==========================================

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
        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return {
            "query":          query,
            "positive_query": parsed.get("positive_query", query),
            "excluded_terms": parsed.get("excluded_terms", []),
            "iso_code":       parsed.get("iso_code", None),
        }
    except Exception as e:
        print(f"Error parsing query '{query}': {e}")
        return {
            "query":          query,
            "positive_query": query,
            "excluded_terms": [],
            "iso_code":       None,
        }


# ==========================================
# BUILD BIO EXAMPLE — Berbeda dari FLAN-T5
# Mengkonversi LLM output ke format BIO token classification
# ==========================================

def word_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    DistilBERT akan re-tokenize dengan WordPiece, tapi label assignment
    dilakukan di level word dulu lalu di-map ke subword saat training.
    """
    return text.split()


def build_bio_example(item: Dict) -> Dict:
    """
    Konversi parsed item ke format BIO untuk DistilBERT token classification.

    Strategy:
    - Tokenize query per kata
    - Cari posisi positive_query dan excluded_terms di dalam query (case-insensitive)
    - Assign label BIO per kata
    - Simpan word_tokens + bio_labels untuk dipakai saat training

    Format output:
    {
        "query"      : "tariffs excluding rice",
        "word_tokens": ["tariffs", "excluding", "rice"],
        "bio_labels" : ["B-POS",  "O",          "B-NEG"],
        "meta": { ...original parsed item... }
    }
    """
    query          = item["query"]
    positive_query = item["positive_query"].strip()
    excluded_terms = item["excluded_terms"]

    query_lower    = query.lower()
    words          = word_tokenize(query)
    labels         = ["O"] * len(words)

    # ── Tag positive query ────────────────────────────────────────────
    pos_lower = positive_query.lower()
    pos_words = word_tokenize(pos_lower)

    if pos_words:
        # Cari urutan kata positive_query di dalam query (sliding window)
        for i in range(len(words) - len(pos_words) + 1):
            window = [w.lower().strip(".,;:!?()[]{}\"'") for w in words[i:i + len(pos_words)]]
            if window == pos_words:
                labels[i] = "B-POS"
                for j in range(1, len(pos_words)):
                    labels[i + j] = "I-POS"
                break

    # ── Tag excluded terms ────────────────────────────────────────────
    for term in excluded_terms:
        term_lower = term.lower().strip()
        term_words = word_tokenize(term_lower)

        if not term_words:
            continue

        for i in range(len(words) - len(term_words) + 1):
            window = [w.lower().strip(".,;:!?()[]{}\"'") for w in words[i:i + len(term_words)]]
            if window == term_words:
                # Jangan override POS label yang sudah ada
                if labels[i] == "O":
                    labels[i] = "B-NEG"
                    for j in range(1, len(term_words)):
                        if labels[i + j] == "O":
                            labels[i + j] = "I-NEG"
                break

    return {
        "query":       query,
        "word_tokens": words,
        "bio_labels":  labels,
        "positive_query": positive_query,
        "excluded_terms": excluded_terms,
        "iso_code":    item.get("iso_code"),
        # meta disimpan terpisah, tidak ikut collate saat training
        "meta": item,
    }


def validate_bio_example(example: Dict) -> bool:
    """
    Validasi kualitas BIO label:
    - Harus ada minimal 1 B-POS
    - Jika query punya negasi, harus ada minimal 1 B-NEG
    - Jumlah token dan label harus sama
    """
    words  = example["word_tokens"]
    labels = example["bio_labels"]

    if len(words) != len(labels):
        return False

    has_pos = any(l == "B-POS" for l in labels)
    has_neg = any(l == "B-NEG" for l in labels)

    negation_words = [
        "excluding", "without", "not", "except", "aside", "other",
        "leaving", "omitted", "skip", "ignore", "minus", "less",
        "excl.", "w/o", "^",
    ]
    query_lower  = example["query"].lower()
    has_negation = any(w in query_lower for w in negation_words)

    if not has_pos:
        return False
    if has_negation and example["excluded_terms"] and not has_neg:
        return False

    return True


def clean_dataset(data: List[Dict]) -> List[Dict]:
    cleaned      = []
    seen_queries = set()

    negation_words = [
        "excluding", "without", "not including", "except", "except for",
        "not related", "not covering", "not about", "but not",
        "aside from", "other than", "leaving out", "omitted",
    ]

    for item in data:
        query    = item["query"]
        positive = item["positive_query"]
        excluded = item["excluded_terms"]

        if query.lower() in seen_queries:
            continue
        seen_queries.add(query.lower())

        has_negation = any(word in query.lower() for word in negation_words)

        if has_negation and not excluded:
            print(f"Filtering (failed parsing): {query}")
            continue

        if excluded:
            positive_lower      = positive.lower()
            still_has_excluded  = any(term.lower() in positive_lower for term in excluded)
            if still_has_excluded:
                print(f"Filtering (excluded leaked in positive): {query}")
                continue

        cleaned.append(item)

    return cleaned


def generate_dataset(
    num_samples: int = 400,
    output_file: str = "synthetic_dataset_distilbert.json",
    model: str       = "gpt-4o-mini",
):
    print(f"Generating {num_samples} samples using {model} (DistilBERT BIO format)...")

    dataset  = []
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

    # Konversi ke format BIO
    bio_data   = [build_bio_example(item) for item in cleaned]

    # Validasi kualitas BIO label
    valid_data = [ex for ex in bio_data if validate_bio_example(ex)]
    skipped    = len(bio_data) - len(valid_data)

    print(f"\nOriginal  : {len(dataset)} samples")
    print(f"Cleaned   : {len(cleaned)} samples")
    print(f"BIO valid : {len(valid_data)} samples")
    if skipped > 0:
        print(f"Skipped   : {skipped} samples (BIO label tidak valid)")

    # Preview 3 sample untuk verifikasi
    print("\n=== Preview 3 sample BIO ===")
    for i, ex in enumerate(valid_data[:3]):
        print(f"\n[{i+1}] Query : {ex['query']}")
        for word, label in zip(ex["word_tokens"], ex["bio_labels"]):
            print(f"       {label:<8} {word}")

    with open(output_file, "w") as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(valid_data)} samples to {output_file}")
    return valid_data


if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    output_file = sys.argv[2] if len(sys.argv) > 2 else "synthetic_dataset_distilbert.json"
    generate_dataset(num_samples=num_samples, output_file=output_file)