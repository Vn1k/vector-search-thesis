import sys
import time
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
import pycountry

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "document_rtas"
MODEL_NAME = "all-mpnet-base-v2"
VECTOR_SIZE = 768
DATA_LIMIT = None
FILE_COVER = "assets/dss_trade_agreements_202511042008.csv"
FILE_DETAIL = "assets/dss_trade_agreement_detail_formatted_202511042008.csv"

def init_qdrant():
    print("⏳ Menghubungkan ke Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        client.get_collections()
        print("✅ Terhubung ke Qdrant!")
    except Exception as e:
        print(f"❌ GAGAL Connect ke Qdrant. Pastikan Podman/Docker jalan.\nError: {e}")
        sys.exit(1)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"♻️  Collection lama '{COLLECTION_NAME}' di-reset.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' siap menerima data.\n")

    print("⚙️  Membuat Index Filter...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="iso", 
        field_schema="keyword"
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="Agreement", 
        field_schema="keyword"
    )
    print("✅ Indexing Selesai.\n")
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

    print(f"total data before cleanig: {len(df)}")

    df = df.drop_duplicates(subset=['article_text', 'Agreement'])

    print(f"total data after cleaning: {len(df)}")
    
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
        agreement = row['Agreement']
        chapter_name = row['chapter_name'] if pd.notna(row['chapter_name']) else "General Provisions"
        iso = str(row['iso']).split(';') if pd.notna(row['iso']) else []

        text_to_vector = f"{agreement} - {chapter_name} - {judul}: {isi}" #contextual embedding
        vector = model.encode(text_to_vector[:1000], normalize_embeddings=True).tolist() #dense vector generation

        payload = {
            "article_name": judul,
            "article_text": isi,
            "Agreement": agreement,
            "bab": chapter_name,
            "iso": iso
        }

        points.append(PointStruct(id=index, vector=vector, payload=payload))

        if len(points) % 10 == 0:
            print(f"   ... Sedang memproses data ke-{len(points)}")

    print(f"\n📦 Mengupload {len(points)} vectors ke database...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    duration = time.time() - start_time
    print(f"✨ SELESAI! {len(points)} data berhasil masuk dalam {duration:.2f} detik.\n")

def find_document(client, model, query, limit=5):
    """Fungsi pencarian dokumen."""
    print("-" * 50)
    print(f"🔍 User query: '{query}'")

    detected_iso = detect_iso_from_query(query)
    qdrant_filter = None
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    if detected_iso:
        print(f"ML detect country: {detected_iso}, using filter")

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="iso",
                    match=MatchValue(value=detected_iso)
                )
            ]
        )
    else:
        print("No detectec country, using global")

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    #Retrieval tahap 1
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,  
        query_filter=qdrant_filter,
        limit=20,
        with_payload=True
    ).points

    if not result:
        print("❌ Tidak ditemukan hasil yang relevan.")
        return
    
    print(f"Melakukan Reranking pada {len(result)} candidate")

    cross_inp = []

    for hit in result:
        pl = hit.payload
        context_text = f"{pl['Agreement']} - {pl['bab']} - {pl['article_name']}: {pl['article_text']}"
        cross_inp.append([query, context_text])

    # reranker tahap 2
    cross_scores = reranker.predict(cross_inp)

    ranked_hits = []
    for idx, hit in enumerate(result):
        ranked_hits.append({
            "original_hit": hit,
            "new_score": cross_scores[idx]
        })
    
    ranked_hits = sorted(ranked_hits, key=lambda x: x['new_score'], reverse=True)

    print(f"   ✅ Menampilkan {limit} hasil terbaik setelah Re-ranking:")
    for i in range(min(limit, len(ranked_hits))):
        item = ranked_hits[i]
        new_score = item['new_score']
        original_score = item['original_hit'].score
        pl = item['original_hit'].payload
        
        judul = pl.get('article_name', '')
        janji = pl.get('Agreement', '-')
        iso = pl.get('iso', '')
        teks_preview = pl.get('article_text', '').replace('\n', ' ')

        print(f"\n   [{i+1}] Re-Rank Score: {new_score:.4f} (Vector Score: {original_score:.4f})")
        print(f"       Judul     : {judul}")
        print(f"       Perjanjian: {janji}")
        print(f"       ISO       : {iso}")
        print(f"       Isi       : {teks_preview}...")

def detect_iso_from_query(query):
    query_lower = query.lower()

    manual_map = {
        "russia": "RUS",           
        "vietnam": "VNM",          
        "usa": "USA",              
        "united states": "USA",
        "uk": "GBR",               
        "england": "GBR",
        "south korea": "KOR",      
        "korea": "KOR",
        "brunei": "BRN",           
        "laos": "LAO"             
    }

    for name, code in manual_map.items():
        if name in query_lower:
            return code

    for country in pycountry.countries:
        country_name = country.name.lower()

        if country_name in query_lower:
            return country.alpha_3
        
    return None

if __name__ == "__main__":
    # 1. Inisialisasi
    client = init_qdrant()
    model = SentenceTransformer(MODEL_NAME)

    # 2. Persiapan Data
    df_data = load_data()

    # 3. Proses Embedding & Upload
    process_and_upload(client, model, df_data)

    # 4. Testing Pencarian (Demo)
    find_document(client, model, "Rules of origin criteria for textile products")