"""
Rekonstruksi teks embedding SESUNGGUHNYA sesuai format indexing.py line 227:
    f"{agreement_name} - {chapter_name} - {article_name}: {article_text}"
Termasuk truncation 384 token mpnet-base-v2.
"""
import csv, sys, os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

csv.field_size_limit(sys.maxsize)

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
MAX_TOKENS = tokenizer.model_max_length  # 384

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Ground truth - System.csv")

targets = [
    ("509", "ARTICLE 220", "UK-EU Art. 220 (Rank 3, Rel=3)", 0.8659),
    ("316", "ARTICLE 15.1", "US-Morocco Art. 15.1 (Rank 5, Rel=0)", 0.6998),
    ("176", "ARTICLE 18.71", "CPTPP Art. 18.71 (Rank 14, Rel=3)", 0.6674),
]

# Baca dari CSV
articles = {}
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["query"] != "intellectual property protection without patents":
            continue
        key = (row["rta_id"], row["article_number"])
        if key not in articles:
            articles[key] = row

query = "intellectual property protection without patents"
q_vec = model.encode([query])

print("=" * 80)
print("TEKS YANG SESUNGGUHNYA DI-EMBED KE QDRANT")
print("  Format: '<agreement> - <chapter> - <article>: <article_text>'")
print(f"  Truncation: {MAX_TOKENS} tokens (mpnet-base-v2)")
print("=" * 80)

all_results = []

for rta_id, art_num, label, system_score in targets:
    row = articles.get((rta_id, art_num))
    if not row:
        print(f"\n[NOT FOUND] {label}")
        continue

    agreement = row["agreement_title"]
    chapter = row["chapter_title"]
    art_title = row.get("article_title", "")
    article = f"{row['article_number']} {art_title}".strip()
    article_text = row["article_text"].strip()

    # === Teks yang di-embed (format indexing.py) ===
    text_to_vector = f"{agreement} - {chapter} - {article}: {article_text}"

    # Tokenisasi & truncation
    tokens = tokenizer.encode(text_to_vector)
    total_tokens = len(tokens)
    truncated = total_tokens > MAX_TOKENS

    if truncated:
        kept_tokens = tokens[:MAX_TOKENS]
        text_for_encoding = tokenizer.decode(kept_tokens, skip_special_tokens=True)
    else:
        text_for_encoding = text_to_vector

    # Embed
    text_vec = model.encode([text_for_encoding])
    emb_sim = cosine_similarity(q_vec, text_vec)[0][0]

    # === Hanya article_text (tanpa prefix) ===
    art_only_vec = model.encode([article_text])
    art_only_sim = cosine_similarity(q_vec, art_only_vec)[0][0]

    # Prefix analysis
    prefix = f"{agreement} - {chapter} - {article}: "
    prefix_tokens = len(tokenizer.encode(prefix))
    article_tokens_in = min(MAX_TOKENS - prefix_tokens, total_tokens - prefix_tokens) if truncated else total_tokens - prefix_tokens
    article_tokens_out = max(0, total_tokens - MAX_TOKENS)

    print(f"\n{'─'*80}")
    print(f"  {label}")
    print(f"  Agreement : {agreement}")
    print(f"  Chapter   : {chapter}")
    print(f"  Article   : {article}")
    print(f"  {'─'*70}")
    print(f"  Similarity (ARTIKEL SAJA)       : {art_only_sim:.4f}")
    print(f"  Similarity (FORMAT EMBEDDING)   : {emb_sim:.4f}")
    print(f"  Similarity (SKOR SISTEM)        : {system_score:.4f}")
    print(f"  {'─'*70}")
    print(f"  Total chars  : {len(text_to_vector)}")
    print(f"  Total tokens : {total_tokens}")
    print(f"  Truncated    : {truncated}")
    if truncated:
        print(f"  Prefix tokens         : {prefix_tokens}")
        print(f"  Article tokens masuk  : {article_tokens_in}")
        print(f"  Article tokens dibuang: {article_tokens_out}")
    print(f"  {'─'*70}")
    print(f"  200 karakter PERTAMA teks yang di-embed:")
    print(f"  \"{text_for_encoding[:200]}...\"")

    all_results.append((label, art_only_sim, emb_sim, system_score))

print(f"\n{'='*80}")
print("RINGKASAN")
print(f"{'='*80}")
print(f"  {'Artikel':<32} {'ArtSaja':>8} {'EmbedFmt':>8} {'System':>8} {'Delta(E-S)':>10}")
print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
for label, art, emb, sys_score in all_results:
    delta = emb - sys_score
    print(f"  {label:<32} {art:>8.4f} {emb:>8.4f} {sys_score:>8.4f} {delta:>+10.4f}")
