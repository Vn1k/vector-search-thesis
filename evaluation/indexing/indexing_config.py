import sys
import time
import pandas as pd
import csv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct, 
    HnswConfigDiff, 
    CollectionStatus
)
from sentence_transformers import SentenceTransformer
from src.config import settings
import localspelling

# ==========================================
# CONFIG GRID SEARCH
# ==========================================
HNSW_GRID_CONFIGS = [
    {"m": 8,  "ef_construct": 100, "collection_name": "rtas_hybrid_m8_ef100"},
    {"m": 8,  "ef_construct": 200, "collection_name": "rtas_hybrid_m8_ef200"},
    {"m": 8,  "ef_construct": 400, "collection_name": "rtas_hybrid_m8_ef400"},
    
    {"m": 16, "ef_construct": 100, "collection_name": "rtas_hybrid_m16_ef100"}, # <-- DEFAULT QDRANT
    {"m": 16, "ef_construct": 200, "collection_name": "rtas_hybrid_m16_ef200"},
    {"m": 16, "ef_construct": 400, "collection_name": "rtas_hybrid_m16_ef400"},
    
    {"m": 32, "ef_construct": 100, "collection_name": "rtas_hybrid_m32_ef100"},
    {"m": 32, "ef_construct": 200, "collection_name": "rtas_hybrid_m32_ef200"},
    {"m": 32, "ef_construct": 400, "collection_name": "rtas_hybrid_m32_ef400"},
]


def load_data():
    """Membaca hybrid chunks dari CSV."""
    print(f"📂 Membaca hybrid chunks dari {settings.FILE_CHUNKS}...")
    try:
        df = pd.read_csv(settings.FILE_CHUNKS, low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ File tidak ditemukan: {e}")
        sys.exit(1)

    df['chunk_index'] = df.groupby('row_index').cumcount()
    chunk_counts      = df.groupby('row_index').size().rename('total_chunks')
    df                = df.merge(chunk_counts, on='row_index', how='left')

    if settings.DATA_LIMIT:
        df = df.head(settings.DATA_LIMIT)

    print(f"📊 Total chunks siap diproses : {len(df)}")
    return df


def prepare_points(model, df):
    """
    Melakukan dialect normalization dan embedding SEMUA data SATU KALI.
    Mengembalikan list berisi PointStruct siap upload untuk menghemat waktu.
    """
    points = []
    changes_count = 0
    processed_count = 0

    print("\n🚀 [FASE 1] Memulai proses Embedding ke Memory (Hanya 1x jalan)...")
    start_time = time.time()

    for idx, row in df.iterrows():
        # ── Setup Payload ─────────────────────────────────────────────────
        raw_display    = row.get('article_text_display', None)
        final_display  = None
        if pd.notna(raw_display):
            s = str(raw_display).strip()
            if not (s.lower().startswith('http') or s.lower().startswith('<a')):
                final_display = s

        payload = {
            "rta_id"              : int(row['rta_id']),
            "article_id"          : row['article_id'],
            "article_number"      : row['article_number'] if pd.notna(row.get('article_number')) else "",
            "article_name"        : row['article_name'] if pd.notna(row.get('article_name')) else "",
            "article_text"        : row['article_text'],
            "article_text_display": final_display,
            "Agreement"           : row['Agreement'],
            "AgreementType"       : [t.strip() for t in str(row['AgreementType']).split(';')] if pd.notna(row.get('AgreementType')) else ["Trade Agreement"],
            "chapter_id"          : row['chapter_id'],
            "chapter_number"      : row['chapter_number'] if pd.notna(row.get('chapter_number')) else "",
            "chapter_name"        : row['chapter_name'] if pd.notna(row.get('chapter_name')) else "General Provisions",
            "iso"                 : str(row['iso']).split(';') if pd.notna(row.get('iso')) else [],
            "acronym"             : row['acronym'] if pd.notna(row.get('acronym')) else "",
            "region"              : [r.strip() for r in str(row.get('region', 'Global')).split(';')],
            "chunk_index"         : int(row['chunk_index']),
            "total_chunks"        : int(row['total_chunks']),
            "chunk_method"        : str(row['method_used']),
            "chunk_tokens"        : int(row['chunk_tokens']),
        }

        # ── Proses Teks & Embedding ───────────────────────────────────────
        chunk_text = str(row['chunk_text'])
        chunk_for_embedding = localspelling.convert_spelling(chunk_text, "us")

        if chunk_text != chunk_for_embedding:
            changes_count += 1

        vector = model.encode(chunk_for_embedding, normalize_embeddings=True).tolist()
        points.append(PointStruct(id=processed_count, vector=vector, payload=payload))
        processed_count += 1

        if processed_count % 10000 == 0:
            print(f"   ⏳ Embedding {processed_count}/{len(df)} data...")

    duration = time.time() - start_time
    print(f"✅ Embedding Selesai! Waktu: {duration:.2f} detik. Total: {len(points)} points disimpan di RAM.")
    return points


def init_and_upload_grid(client, points):
    """
    Looping melalui HNSW_GRID_CONFIGS, buat collection, upload points, dan tunggu sampai GREEN.
    Waktu dicatat ke CSV.
    """
    print("\n🚀 [FASE 2] Memulai GRID SEARCH Upload (9 Collection)...")
    
    csv_file = "indexing_performance.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["collection_name", "m", "ef_construct", "indexing_time_seconds"])

    batch_size = 200

    for config in HNSW_GRID_CONFIGS:
        col_name = config["collection_name"]
        m = config["m"]
        ef_c = config["ef_construct"]

        print(f"\n" + "="*50)
        print(f"⚙️ Memproses Koleksi: {col_name} (m={m}, ef_construct={ef_c})")
        print("="*50)

        # 1. Reset & Buat Collection
        if client.collection_exists(col_name):
            client.delete_collection(col_name)
            print(f"♻️ Menghapus collection lama '{col_name}'...")

        # MENGGUNAKAN HNSW CONFIG DIFF DI SINI!
        client.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=m, 
                ef_construct=ef_c, 
                full_scan_threshold=10000  # Mengikuti config Anda
            )
        )

        # 2. Buat Index Metadata
        for field, schema in [("iso", "keyword"), ("Agreement", "keyword"), ("rta_id", "integer")]:
            client.create_payload_index(collection_name=col_name, field_name=field, field_schema=schema)

        # 3. Upload Data ke Qdrant
        start_upload = time.time()
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=col_name, points=batch)
            if (i + batch_size) % 20000 == 0:
                print(f"   ⬆️ Upload progress: {i + batch_size}/{len(points)}...")

        # 4. Tunggu HNSW Indeks Selesai (Status GREEN)
        print("   ⏳ Menunggu grafik HNSW selesai dijahit (Status = GREEN)...")
        while True:
            info = client.get_collection(collection_name=col_name)
            if info.status == CollectionStatus.GREEN:
                break
            time.sleep(5)  # Tunggu 5 detik lalu cek lagi

        indexing_time = time.time() - start_upload
        print(f"✅ Indeks {col_name} selesai sepenuhnya dalam {indexing_time:.2f} detik!")

        # 5. Catat hasil ke CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([col_name, m, ef_c, indexing_time])

    print(f"\n🎉 SELURUH GRID SEARCH SELESAI! Laporan disimpan di '{csv_file}'.")


if __name__ == "__main__":
    # 1. Hubungkan ke Qdrant
    print(f"⏳ Menghubungkan ke Qdrant di {settings.QDRANT_HOST}...")
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=60)
    
    # 2. Load Model
    print(f"🧠 Loading Model AI: {settings.VECTOR_MODEL}...")
    model = SentenceTransformer(settings.VECTOR_MODEL)

    # 3. Load DataFrame
    dataframe = load_data()

    # 4. FASE 1: Prepare & Embed
    all_points = prepare_points(model, dataframe)

    # 5. FASE 2: Looping Grid Search
    init_and_upload_grid(client, all_points)