import sys
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from src.config import settings
import localspelling


def tail_preview(text, tail_chars=180):
    """Mengambil potongan akhir teks agar log tetap ringkas."""
    normalized = " ".join(str(text).split())
    if len(normalized) <= tail_chars:
        return normalized
    return f"...{normalized[-tail_chars:]}"

def get_model_tokenizer(model):
    """Ambil tokenizer dari SentenceTransformer."""
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    first_module = model._first_module()
    if hasattr(first_module, "tokenizer"):
        return first_module.tokenizer
    raise RuntimeError("Tokenizer tidak ditemukan pada SentenceTransformer model.")


def print_log_snapshot(start_row, end_row, last_spelling_log, is_final=False):
    """Tampilkan ringkasan log untuk rentang data tertentu."""
    label = "LOG SNAPSHOT AKHIR" if is_final else "LOG SNAPSHOT"
    print("\n" + "=" * 30)
    print(f"{label} DATA {start_row}-{end_row}")

    if last_spelling_log:
        print(f"[SPELLING] Perubahan terakhir UK -> US pada baris {last_spelling_log['row_index']}:")
        for original, converted in last_spelling_log["changes"]:
            print(f"   - {original} -> {converted}")
        if last_spelling_log["extra_changes"] > 0:
            print(f"   - ... dan {last_spelling_log['extra_changes']} perubahan lain")
    else:
        print("[SPELLING] Tidak ada perubahan spelling pada rentang ini.")

    print("=" * 30)


def init_qdrant():
    print(f"⏳ Menghubungkan ke Qdrant di {settings.QDRANT_HOST}...")
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    try:
        client.get_collections()
        print("✅ Terhubung ke Qdrant!")
    except Exception as e:
        print(f"❌ GAGAL Connect ke Qdrant. Pastikan Podman/Docker jalan.\nError: {e}")
        sys.exit(1)

    if client.collection_exists(settings.COLLECTION_NAME_RECURSIVE):
        client.delete_collection(settings.COLLECTION_NAME_RECURSIVE)
        print(f"♻️  Collection lama '{settings.COLLECTION_NAME_RECURSIVE}' di-reset.")

    client.create_collection(
        collection_name=settings.COLLECTION_NAME_RECURSIVE,
        vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{settings.COLLECTION_NAME_RECURSIVE}' siap menerima data.\n")

    print("⚙️  Membuat Index Filter...")
    for field, schema in [
        ("iso",           "keyword"),
        ("Agreement",     "keyword"),
        ("AgreementType", "keyword"),
        ("rta_id",        "integer"),
        ("chunk_index",   "integer"),
        ("article_id",    "keyword"),
        ("article_number","keyword"),
        ("chapter_id",    "keyword"),
        ("chapter_number","keyword")
    ]:
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME_RECURSIVE,
            field_name=field,
            field_schema=schema,
        )
    print("✅ Indexing Selesai.\n")
    return client


def load_data():
    """
    Membaca hybrid_chunks_result.csv yang sudah mengandung semua metadata
    (rta_id, iso, acronym, AgreementType, region, article_text_display).
    File ini dihasilkan dari hybrid chunking notebook di Colab.
    """
    print(f"📂 Membaca hybrid chunks dari {settings.FILE_CHUNKS_RECURSIVE}...")
    try:
        df = pd.read_csv(settings.FILE_CHUNKS_RECURSIVE, low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e}")
        sys.exit(1)

    # Hitung chunk_index dan total_chunks per dokumen
    df['chunk_index'] = df.groupby('row_index').cumcount()
    chunk_counts      = df.groupby('row_index').size().rename('total_chunks')
    df                = df.merge(chunk_counts, on='row_index', how='left')

    if settings.DATA_LIMIT:
        df = df.head(settings.DATA_LIMIT)

    print(f"📊 Total chunks siap diproses : {len(df)}")
    print(f"   Dokumen unik               : {df['row_index'].nunique()}")
    print(f"   Jalur Semantic             : {len(df[df['method_used'] == 'Semantic'])}")
    print(f"   Jalur Recursive-Fallback   : {len(df[df['method_used'] == 'Recursive-Fallback'])}")
    print()
    return df


def process_and_upload(client, model, df):
    """
    Embed setiap chunk dan upload ke Qdrant.

    Yang di-embed  : chunk_text (teks potongan, sudah dalam batas token)
    Yang disimpan  : article_text FULL dari original CSV (untuk display ke user)

    Dengan cara ini:
    - Search menemukan chunk yang paling relevan
    - User melihat artikel lengkap, bukan potongan
    - Grouping by rta_id di search_service tetap bekerja normal
    """
    points               = []
    batch_size           = 100
    log_every            = 1000
    changes_count        = 0
    processed_count      = 0
    current_window_start = 1
    last_spelling_log    = None

    print("🚀 Memulai proses Embedding (Hybrid Chunking)...")
    print(f"⚙️  Token limit model : {model.get_max_seq_length() or 384}")
    start_time = time.time()

    for idx, row in df.iterrows():
        # ── Metadata dokumen ──────────────────────────────────────────────
        article_id     = row['article_id']
        article_number = row['article_number'] if pd.notna(row.get('article_number'))  else ""
        article_name   = row['article_name']  if pd.notna(row.get('article_name'))  else ""
        article_text   = row['article_text']
        agreement_name = row['Agreement']
        chapter_id     = row['chapter_id']
        chapter_number = row['chapter_number'] if pd.notna(row.get('chapter_number'))  else ""
        chapter_name   = row['chapter_name']  if pd.notna(row.get('chapter_name'))  else "General Provisions"
        acronym        = row['acronym']        if pd.notna(row.get('acronym'))        else ""
        iso            = str(row['iso']).split(';') if pd.notna(row.get('iso')) else []

        raw_display    = row.get('article_text_display', None)
        final_display  = None
        if pd.notna(raw_display):
            s = str(raw_display).strip()
            if not (s.lower().startswith('http') or s.lower().startswith('<a')):
                final_display = s

        raw_type       = str(row['AgreementType']) if pd.notna(row.get('AgreementType')) else "Trade Agreement"
        agreement_type = [t.strip() for t in raw_type.split(';')]

        raw_region     = row['region'] if pd.notna(row.get('region')) else "Global"
        region         = [r.strip() for r in raw_region.split(';')]

        # ── Chunk metadata ────────────────────────────────────────────────
        chunk_text    = str(row['chunk_text'])
        chunk_index   = int(row['chunk_index'])
        total_chunks  = int(row['total_chunks'])
        method_used   = str(row['method_used'])
        chunk_tokens  = int(row['chunk_tokens'])

        # ── Dialect normalization pada chunk_text (untuk embedding) ───────
        chunk_for_embedding = localspelling.convert_spelling(chunk_text, "us")

        if chunk_text != chunk_for_embedding:
            changes_count += 1
            orig_words = chunk_text.split()
            conv_words = chunk_for_embedding.split()
            spelling_changes = [
                (orig_words[i], conv_words[i])
                for i in range(min(len(orig_words), len(conv_words)))
                if orig_words[i] != conv_words[i]
            ]
            if spelling_changes:
                last_spelling_log = {
                    "row_index": idx,
                    "changes"  : spelling_changes[:8],
                    "extra_changes": max(0, len(spelling_changes) - 8),
                }

        # ── Embed chunk ───────────────────────────────────────────────────
        # Tidak perlu truncation — hybrid pipeline sudah menjamin ≤ token limit
        vector = model.encode(chunk_for_embedding, normalize_embeddings=True).tolist()

        # ── Payload: article_text FULL untuk display ──────────────────────
        payload = {
            "rta_id"              : int(row['rta_id']),
            "article_id"          : article_id,
            "article_number"      : article_number,
            "article_name"        : article_name,
            "article_text"        : article_text,          # ← FULL untuk display
            "article_text_display": final_display,
            "Agreement"           : agreement_name,
            "AgreementType"       : agreement_type,
            "chapter_id"          : chapter_id,
            "chapter_number"      : chapter_number,
            "chapter_name"        : chapter_name,
            "iso"                 : iso,
            "acronym"             : acronym,
            "region"              : region,
            # ── Chunk metadata (untuk analisis/debug) ──────────────────
            "chunk_index"         : chunk_index,
            "total_chunks"        : total_chunks,
            "chunk_method"        : method_used,
            "chunk_tokens"        : chunk_tokens,
        }

        points.append(PointStruct(id=processed_count, vector=vector, payload=payload))
        processed_count += 1

        if len(points) >= batch_size:
            client.upsert(collection_name=settings.COLLECTION_NAME_RECURSIVE, points=points)
            print(f"   ✅ Upload {len(points)} chunks (Total: {processed_count})")
            points = []

        if processed_count % log_every == 0:
            print_log_snapshot(
                start_row=current_window_start,
                end_row=processed_count,
                last_spelling_log=last_spelling_log,
            )
            current_window_start = processed_count + 1
            last_spelling_log    = None

    # Upload sisa
    if points:
        client.upsert(collection_name=settings.COLLECTION_NAME_RECURSIVE, points=points)
        print(f"   ✅ Upload sisa {len(points)} chunks.")

    if processed_count % log_every != 0:
        print_log_snapshot(
            start_row=current_window_start,
            end_row=processed_count,
            last_spelling_log=last_spelling_log,
            is_final=True,
        )

    duration = time.time() - start_time
    print("\n" + "=" * 40)
    print(f"✨ PROSES SELESAI!")
    print(f"📦 Total chunks diupload       : {processed_count}")
    print(f"🔤 Chunks dengan spelling fix  : {changes_count}")
    print(f"⏱️  Waktu total                : {duration:.2f} detik ({duration/60:.1f} menit)")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    # 1. Setup Database
    qdrant = init_qdrant()

    # 2. Load Model AI
    print(f"🧠 Loading Model AI: {settings.VECTOR_MODEL}...")
    model = SentenceTransformer(settings.VECTOR_MODEL)

    # 3. Load chunks
    dataframe = load_data()

    # 4. Embed & Upload
    process_and_upload(qdrant, model, dataframe)