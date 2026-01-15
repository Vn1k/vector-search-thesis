import sys
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "document_rtas"
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_LIMIT = 100
FILE_COVER = "assets/dss_trade_agreements_202511042008.csv"
FILE_DETAIL = "assets/dss_trade_agreement_detail_formatted_202511042008.csv"

def init_qdrant():
    """Menghubungkan ke Qdrant dan menyiapkan Collection."""
    print("⏳ Menghubungkan ke Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        client.get_collections()
        print("✅ Terhubung ke Qdrant!")
    except Exception as e:
        print(f"❌ GAGAL Connect ke Qdrant. Pastikan Podman/Docker jalan.\nError: {e}")
        sys.exit(1)

    # Reset Collection Lama
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"♻️  Collection lama '{COLLECTION_NAME}' di-reset.")

    # Buat Collection Baru
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' siap menerima data.\n")
    return client

def load_data():
    """Membaca dan membersihkan data CSV."""
    print(f"📂 Membaca file CSV dari {FILE_DETAIL}...")
    dt_cover = pd.read_csv(FILE_COVER)
    dt_detail = pd.read_csv(FILE_DETAIL)
    
    # Merge Data
    df = pd.merge(dt_detail, dt_cover, left_on='rta_id', right_on='id', how='inner')
    
    # Hapus baris yang teks pasalnya kosong
    df = df.dropna(subset=['article_text'])
    
    # Batasi jumlah data jika diperlukan
    if DATA_LIMIT:
        df = df.head(DATA_LIMIT)
        
    print(f"📊 Total Data Siap Diproses: {len(df)} baris.\n")
    return df

def process_and_upload(client, model, df):
    """Melakukan embedding dan upload data ke Qdrant."""
    points = []
    print("🚀 Memulai proses Embedding...")
    start_time = time.time()

    for index, row in df.iterrows():
        judul = row['article_name'] if pd.notna(row['article_name']) else ""
        isi = row['article_text']
        
        # Format: "Judul: Isi" (atau cuma Isi jika judul kosong)
        text_to_vector = f"{judul}: {isi}" if judul else isi

        agreement = row['Agreement']
        bab = row['chapter_name'] if pd.notna(row['chapter_name']) else "General"
        # Split ISO "IDN;MYS" -> ["IDN", "MYS"]
        iso = str(row['iso']).split(';') if pd.notna(row['iso']) else []

        # Embedding (Vectorize)
        vector = model.encode(text_to_vector[:1000], normalize_embeddings=True).tolist()

        # Bungkus Payload
        payload = {
            "article_name": judul,
            "article_text": isi,
            "Agreement": agreement,
            "bab": bab,
            "iso": iso
        }

        points.append(PointStruct(id=index, vector=vector, payload=payload))

        # Print progress sederhana biar terminal gak penuh
        if len(points) % 10 == 0:
            print(f"   ... Sedang memproses data ke-{len(points)}")

    # Upload ke Qdrant
    print(f"\n📦 Mengupload {len(points)} vectors ke database...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    duration = time.time() - start_time
    print(f"✨ SELESAI! {len(points)} data berhasil masuk dalam {duration:.2f} detik.\n")

def find_document(client, model, query, limit=3):
    """Fungsi pencarian dokumen."""
    print("-" * 50)
    print(f"🔍 User query: '{query}'")

    # Encode query user
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,  
        limit=limit,
        with_payload=True
    ).points

    if not result:
        print("❌ Tidak ditemukan hasil yang relevan.")
        return

    print(f"   Ditemukan {len(result)} hasil relevan:")
    for i, hit in enumerate(result):
        score = hit.score
        payload = hit.payload
        
        judul = payload.get('article_name', '(Tanpa Judul)')
        janji = payload.get('Agreement', '-')
        teks_preview = payload.get('article_text', '')[:150].replace('\n', ' ')

        print(f"\n   [{i+1}] Score: {score:.4f}")
        print(f"       Judul     : {judul}")
        print(f"       Perjanjian: {janji}")
        print(f"       Isi       : {teks_preview}...")

if __name__ == "__main__":
    # 1. Inisialisasi
    client = init_qdrant()
    model = SentenceTransformer(MODEL_NAME)

    # 2. Persiapan Data
    df_data = load_data()

    # 3. Proses Embedding & Upload
    process_and_upload(client, model, df_data)

    # 4. Testing Pencarian (Demo)
    find_document(client, model, "rules of origin certificate")