"""
Jalankan script ini di environment lokalmu yang sudah ada model mpnet-v2.
Pastikan: pip install sentence-transformers scikit-learn
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("=" * 65)
print("TABEL 1: Similarity kata kunci vs 'protection'")
print("=" * 65)

anchor = "protection"
words = [
    "protection",
    "enforcement",
    "infringement",
    "remedies",
    "patents",
    "rights",
    "intellectual property",
]

vecs = model.encode([anchor] + words)
anchor_vec = vecs[0:1]
for word, vec in zip(words, vecs[1:]):
    sim = cosine_similarity(anchor_vec, vec.reshape(1, -1))[0][0]
    print(f"  {word:<30} {sim:.4f}")


print()
print("=" * 65)
print("TABEL 2a: Similarity kueri vs SNIPPET (skrip asli Claude)")
print("=" * 65)

query = "intellectual property protection without patents"

snippets = {
    "UK-EU Art. 220 (Rank 3, Rel=3)":
        "this title shall complement protection and enforcement of "
        "intellectual property rights does not preclude party from "
        "introducing more extensive protection",
    "US-Morocco Art. 15.1 (Rank 5, Rel=0)":
        "each party shall ratify or accede to the patent cooperation "
        "treaty the budapest treaty for the purposes of patent procedure",
    "CPTPP Art. 18.71 (Rank 14, Rel=3)":
        "each party shall ensure that enforcement procedures are available "
        "to permit effective action against any act of infringement of "
        "intellectual property rights including expeditious remedies to "
        "prevent infringements",
}

q_vec = model.encode([query])
for label, sent in snippets.items():
    s_vec = model.encode([sent])
    sim = cosine_similarity(q_vec, s_vec)[0][0]
    print(f"  {sim:.4f}  |  {label}")


print()
print("=" * 65)
print("TABEL 2b: Similarity kueri vs FULL TEXT artikel (dari Ground Truth CSV)")
print("=" * 65)
print("  Catatan: mpnet-base-v2 max 384 token; teks >384 token akan terpotong")
print()

import csv, sys, os
csv.field_size_limit(sys.maxsize)

# Resolve path relatif terhadap lokasi script ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "Ground truth - System.csv")

full_texts = {}
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["query"] != "intellectual property protection without patents":
            continue
        key = (row["rta_id"], row["article_number"])
        if key not in full_texts:
            full_texts[key] = row

targets = [
    ("509", "ARTICLE 220", "UK-EU Art. 220 (Rank 3, Rel=3)", 452),
    ("316", "ARTICLE 15.1", "US-Morocco Art. 15.1 (Rank 5, Rel=0)", 4771),
    ("176", "ARTICLE 18.71", "CPTPP Art. 18.71 (Rank 14, Rel=3)", 2216),
]

for rta_id, article_num, label, text_len in targets:
    row = full_texts.get((rta_id, article_num))
    if row is None:
        print(f"  [NOT FOUND] {label}")
        continue
    text = row["article_text"].strip()
    s_vec = model.encode([text])
    sim = cosine_similarity(q_vec, s_vec)[0][0]
    # Estimate tokens (rough: chars/4)
    est_tokens = len(text) // 4
    trunc_note = " (TRUNCATED by model)" if est_tokens > 384 else ""
    print(f"  {sim:.4f}  |  {label}")
    print(f"           |  {len(text)} chars, ~{est_tokens} tokens{trunc_note}")
    print(f"           |  {row['agreement_title']}")
    print(f"           |  Chapter: {row['chapter_title']}")
    print()


print()
print("=" * 65)
print("TABEL 3: Frasa kueri vs frasa kunci artikel")
print("=" * 65)

query_phrases = [
    "intellectual property protection",
    "without patents",
]
article_phrases = [
    "protection of intellectual property rights",  # ideal match
    "enforcement procedures for intellectual property",  # Art 18.71
    "infringement of intellectual property rights",  # Art 18.71
    "remedies to prevent infringements",  # Art 18.71
    "patent cooperation treaty",  # Art 15.1
    "ratify patent agreements",  # Art 15.1
]

qp_vecs = model.encode(query_phrases)
ap_vecs = model.encode(article_phrases)

header = f"{'Frasa Artikel':<45}"
for qp in query_phrases:
    header += f"  {qp[:22]:<22}"
print(f"  {header}")
print("  " + "-" * 95)

for phrase, av in zip(article_phrases, ap_vecs):
    row = f"  {phrase:<45}"
    for qv in qp_vecs:
        sim = cosine_similarity(qv.reshape(1, -1), av.reshape(1, -1))[0][0]
        row += f"  {sim:.4f}              "
    print(row)