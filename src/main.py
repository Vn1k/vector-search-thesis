from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct
import pandas as pd
import sys

# --- 1. SETUP KONEKSI YANG BENAR (FIX IP) ---
# Gunakan 'host' dan 'port' terpisah, dan pakai IP 127.0.0.1 (Bukan localhost)
# Ini memaksa koneksi lewat IPv4 agar tidak diputus oleh Linux.
client = QdrantClient(host="127.0.0.1", port=6333)

print("⏳ Mengecek koneksi ke Qdrant...")
try:
    # Tes koneksi ringan dulu
    client.get_collections()
    print("✅ BERHASIL: Python terhubung ke Qdrant!")
except Exception as e:
    print(f"❌ GAGAL: Tidak bisa connect. Pastikan Podman jalan. Error: {e}")
    sys.exit(1)

# --- 2. BUAT COLLECTION ---
collection_name = "document_rtas"

# Cek agar tidak error kalau dijalankan berkali-kali
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{collection_name}' dibuat.")
else:
    print(f"ℹ️ Collection '{collection_name}' sudah ada.")

# --- 3. PROSES DATA ---
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading CSV file..")
# Pastikan path file benar
dt_cover = pd.read_csv("assets/dss_trade_agreements_202511042008.csv")
dt_detail = pd.read_csv("assets/dss_trade_agreement_detail_formatted_202511042008.csv")

# --- PERBAIKAN TYPO DISINI ---
# Sebelumnya kamu tulis 'dt_final' tapi dipanggil 'df_final'
df_final = pd.merge(dt_detail, dt_cover, left_on='rta_id', right_on='id', how='inner')

# Sekarang variabelnya sudah konsisten 'df_final'
df_final = df_final.dropna(subset=['article_text'])

# Tampilkan Data
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

print("\n" + "="*50)
print(f"DATA SIAP: {len(df_final)} Baris")
print("="*50)

# Tampilkan kolom penting saja
kolom = ['rta_id', 'Agreement', 'chapter_name', 'article_name', 'article_text', 'iso']
print(df_final[kolom].head(50))