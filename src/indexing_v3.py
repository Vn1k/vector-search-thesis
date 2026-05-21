import sys
import time
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer, SparseEncoder
from src.config import settings
import localspelling


def tail_preview(text, tail_chars=180):
    """Mengambil potongan akhir teks agar log tetap ringkas."""
    normalized = " ".join(str(text).split())
    if len(normalized) <= tail_chars:
        return normalized
    return f"...{normalized[-tail_chars:]}"


def sparse_embedding_to_qdrant_vector(sparse_embedding):
    """Konversi output SparseEncoder ke format SparseVector Qdrant."""
    if hasattr(sparse_embedding, "coalesce"):
        sparse_embedding = sparse_embedding.coalesce()

    if callable(getattr(sparse_embedding, "indices", None)):
        tensor_indices = sparse_embedding.indices()
        tensor_values = sparse_embedding.values()

        if tensor_indices.dim() == 2:
            vector_indices = tensor_indices[-1]
        else:
            vector_indices = tensor_indices

        indices = [int(index) for index in vector_indices.cpu().tolist()]
        values = [float(value) for value in tensor_values.cpu().tolist()]
        return indices, values

    indices = sparse_embedding.indices.tolist()
    values = sparse_embedding.values.tolist()
    return [int(index) for index in indices], [float(value) for value in values]


def get_model_tokenizer(model):
    """Ambil tokenizer dari SentenceTransformer."""
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    first_module = model._first_module()
    if hasattr(first_module, "tokenizer"):
        return first_module.tokenizer
    raise RuntimeError("Tokenizer tidak ditemukan pada SentenceTransformer model.")


def print_log_snapshot(start_row, end_row, last_spelling_log, sparse_stats=None, is_final=False):
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

    if sparse_stats and sparse_stats["count"] > 0:
        avg_nonzero = sparse_stats["nonzero_total"] / sparse_stats["count"]
        print(
            f"[SPARSE] Non-zero terms avg/max: {avg_nonzero:.1f}/{sparse_stats['nonzero_max']}"
        )
        last_sparse = sparse_stats.get("last_sparse")
        if last_sparse:
            print(
                f"[SPARSE] Contoh terakhir baris {last_sparse['row_index']}: "
                f"{last_sparse['nonzero_terms']} terms, max_weight={last_sparse['max_weight']:.4f}"
            )
            print(f"   indices sample: {last_sparse['indices_sample']}")
            print(f"   values sample : {last_sparse['values_sample']}")
    else:
        print("[SPARSE] Belum ada sparse vector pada rentang ini.")

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

    if client.collection_exists(settings.COLLECTION_NAME_WITH_SPARSE):
        client.delete_collection(settings.COLLECTION_NAME_WITH_SPARSE)
        print(f"♻️  Collection lama '{settings.COLLECTION_NAME_WITH_SPARSE}' di-reset.")

    client.create_collection(
        collection_name=settings.COLLECTION_NAME_WITH_SPARSE,
        vectors_config={
            "dense_vector": models.VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_vector": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        }
    )
    print(f"✅ Collection '{settings.COLLECTION_NAME_WITH_SPARSE}' siap menerima data.\n")

    print("⚙️  Membuat Index Filter...")
    for field, schema in [
        ("iso",           "keyword"),
        ("Agreement",     "keyword"),
        ("AgreementType", "keyword"),
        ("rta_id",        "integer"),
        ("chunk_index",   "integer"),
    ]:
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME_WITH_SPARSE,
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
    print(f"📂 Membaca hybrid chunks dari {settings.FILE_CHUNKS}...")
    try:
        df = pd.read_csv(settings.FILE_CHUNKS, low_memory=False)
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


def process_and_upload(client, model_dense, model_sparse, df):
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
    sparse_stats         = {
        "count": 0,
        "nonzero_total": 0,
        "nonzero_max": 0,
        "last_sparse": None,
    }

    print("🚀 Memulai proses Embedding (Hybrid Chunking)...")
    print(f"⚙️  Token limit model : {model_dense.get_max_seq_length() or 384}")
    start_time = time.time()

    for idx, row in df.iterrows():
        # ── Metadata dokumen ──────────────────────────────────────────────
        article_name   = row['article_name']  if pd.notna(row.get('article_name'))  else ""
        # article_text   = row['article_text']
        agreement_name = row['Agreement']
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
        dense = model_dense.encode(chunk_for_embedding, normalize_embeddings=True).tolist()

        sparse = model_sparse.encode_document(
            chunk_for_embedding,
            show_progress_bar=False,
            save_to_cpu=True,
        )
        sparse_indices, sparse_values = sparse_embedding_to_qdrant_vector(sparse)
        sparse_nonzero_terms = len(sparse_indices)
        sparse_stats["count"] += 1
        sparse_stats["nonzero_total"] += sparse_nonzero_terms
        sparse_stats["nonzero_max"] = max(sparse_stats["nonzero_max"], sparse_nonzero_terms)
        sparse_stats["last_sparse"] = {
            "row_index": idx,
            "nonzero_terms": sparse_nonzero_terms,
            "max_weight": max(sparse_values) if sparse_values else 0.0,
            "indices_sample": sparse_indices[:8],
            "values_sample": [round(float(value), 4) for value in sparse_values[:8]],
        }

        # ── Payload: article_text FULL untuk display ──────────────────────
        payload = {
            "rta_id"              : int(row['rta_id']),
            "article_name"        : article_name,
            # "article_text"        : article_text,          # ← FULL untuk display
            "article_text_display": final_display,
            "Agreement"           : agreement_name,
            "AgreementType"       : agreement_type,
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

        points.append(PointStruct(id=processed_count, vector={
            "dense_vector": dense,
            "sparse_vector": models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        }, payload=payload))
        processed_count += 1

        if len(points) >= batch_size:
            client.upsert(collection_name=settings.COLLECTION_NAME_WITH_SPARSE, points=points)
            print(f"   ✅ Upload {len(points)} chunks (Total: {processed_count})")
            points = []

        if processed_count % log_every == 0:
            print_log_snapshot(
                start_row=current_window_start,
                end_row=processed_count,
                last_spelling_log=last_spelling_log,
                sparse_stats=sparse_stats,
            )
            current_window_start = processed_count + 1
            last_spelling_log    = None
            sparse_stats         = {
                "count": 0,
                "nonzero_total": 0,
                "nonzero_max": 0,
                "last_sparse": None,
            }

    # Upload sisa
    if points:
        client.upsert(collection_name=settings.COLLECTION_NAME_WITH_SPARSE, points=points)
        print(f"   ✅ Upload sisa {len(points)} chunks.")

    if processed_count % log_every != 0:
        print_log_snapshot(
            start_row=current_window_start,
            end_row=processed_count,
            last_spelling_log=last_spelling_log,
            sparse_stats=sparse_stats,
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
    print(f"🧠 Loading Dense Model : {settings.VECTOR_MODEL}...")
    dense_model = SentenceTransformer(settings.VECTOR_MODEL)
    print(f"🧠 Loading Sparse Model: {settings.SPARSE_MODEL}...")
    sparse_model = SparseEncoder(settings.SPARSE_MODEL)
    # 3. Load chunks
    dataframe = load_data()

    # 4. Embed & Upload
    process_and_upload(qdrant, dense_model, sparse_model, dataframe)
