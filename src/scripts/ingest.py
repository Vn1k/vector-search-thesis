import sys
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from src.core.config import settings

def init_qdrant():
    print(f"⏳ Menghubungkan ke Qdrant di {settings.QDRANT_HOST}...")
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    try:
        client.get_collections()
        print("✅ Terhubung ke Qdrant!")
    except Exception as e:
        print(f"❌ GAGAL Connect ke Qdrant. Pastikan Podman/Docker jalan.\nError: {e}")
        sys.exit(1)

    if client.collection_exists(settings.COLLECTION_NAME):
        client.delete_collection(settings.COLLECTION_NAME)
        print(f"♻️  Collection lama '{settings.COLLECTION_NAME}' di-reset.")

    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{settings.COLLECTION_NAME}' siap menerima data.\n")

    print("⚙️  Membuat Index Filter...")
    client.create_payload_index(
        collection_name=settings.COLLECTION_NAME,
        field_name="iso", 
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=settings.COLLECTION_NAME,
        field_name="Agreement", 
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=settings.COLLECTION_NAME,
        field_name="AgreementType", 
        field_schema="keyword"
    )
    print("✅ Indexing Selesai.\n")
    return client

def load_data():
    """Membaca dan membersihkan data CSV."""
    print(f"📂 Membaca file CSV dari {settings.FILE_DETAIL}...")
    try:
        dt_cover = pd.read_csv(settings.FILE_COVER)
        dt_detail = pd.read_csv(settings.FILE_DETAIL)
    except FileNotFoundError as e:
        print(f"❌ File CSV tidak ditemukan: {e}")
        sys.exit(1)
    
    # Merge Data
    df = pd.merge(dt_detail, dt_cover, left_on='rta_id', right_on='id', how='inner')
    
    # Hapus baris yang teks pasalnya kosong
    df = df.dropna(subset=['article_text'])

    print(f"total data before cleaning: {len(df)}")

    df = df.drop_duplicates(subset=['article_text', 'Agreement'])

    print(f"total data after cleaning: {len(df)}")
    
    # Batasi jumlah data jika diperlukan
    if settings.DATA_LIMIT:
        df = df.head(settings.DATA_LIMIT)
        
    print(f"📊 Total Data Siap Diproses: {len(df)} baris.\n")
    return df

def process_and_upload(client, model, df):
    """Melakukan embedding dan upload data ke Qdrant."""
    points = []
    batch_size = 100
    print("🚀 Memulai proses Embedding...")
    start_time = time.time()

    for index, row in df.iterrows():
        article_name = row['article_name'] if pd.notna(row['article_name']) else ""
        article_text = row['article_text']
        raw_article_text_display = row['article_text_display']
        final_display_text = article_text

        if pd.notna(raw_article_text_display):
            display_str = str(raw_article_text_display).strip()

            if display_str.lower().startswith('http') or display_str.lower().startswith('<a'):
                final_display_text = article_text
            else:
                final_display_text = display_str
        
        agreement_name = row['Agreement'] # Agreement and title is the same

        raw_agreement_type = str(row['AgreementType']) if pd.notna(row.get('AgreementType')) else "Trade Agreement"
        agreement_type_for_payload =[t.strip() for t in raw_agreement_type.split(';')]

        acronym = row['acronym'] if pd.notna(row.get('acronym')) else ""
        chapter_name = row['chapter_name'] if pd.notna(row['chapter_name']) else "General Provisions"
        iso = str(row['iso']).split(';') if pd.notna(row['iso']) else []

        raw_region = row['region'] if pd.notna(row.get('region')) else "Global"
        region_for_payload = [r.strip() for r in raw_region.split(';')]

        # Contextual Embedding Logic
        text_to_vector = f"{agreement_name} - {chapter_name} - {article_name}: {article_text}"
        
        # Encode
        vector = model.encode(text_to_vector[:1000], normalize_embeddings=True).tolist()

        payload = {
            "article_name": article_name,
            "article_text": article_text,
            "article_text_display": final_display_text,
            "Agreement": agreement_name,
            "AgreementType": agreement_type_for_payload,
            "chapter_name": chapter_name,
            "iso": iso,
            "acronym": acronym,
            "region": region_for_payload
        }

        points.append(PointStruct(id=index, vector=vector, payload=payload))

        if len(points) >= batch_size:
            client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points
            )
            print(f"   ✅ Berhasil upload {len(points)} data (Total terproses: {index + 1})")
            points = []
    
    if points:
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        print(f"   ✅ Berhasil upload sisa {len(points)} data.")
    
    duration = time.time() - start_time
    print(f"✨ SELESAI! {len(points)} data berhasil masuk dalam {duration:.2f} detik.\n")

if __name__ == "__main__":
    # 1. Setup Database
    qdrant = init_qdrant()
    
    # 2. Load Model AI
    print(f"🧠 Loading Model AI: {settings.VECTOR_MODEL}...")
    model = SentenceTransformer(settings.VECTOR_MODEL)
    
    # 3. Baca Data CSV
    dataframe = load_data()
    
    # 4. Masukkan ke Database
    process_and_upload(qdrant, model, dataframe)