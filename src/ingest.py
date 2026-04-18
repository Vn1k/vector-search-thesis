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


def truncate_text_by_token_limit(text, tokenizer, token_limit):
    """Potong teks sesuai batas token model, kembalikan teks hasil potong + statistik token."""
    full_tokenized = tokenizer(text, add_special_tokens=True, truncation=False)
    total_tokens = len(full_tokenized["input_ids"])

    truncated = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=token_limit,
        return_offsets_mapping=getattr(tokenizer, "is_fast", False),
    )
    used_tokens = len(truncated["input_ids"])

    if total_tokens <= token_limit:
        return text, total_tokens, used_tokens

    cut_char_idx = len(text)
    if "offset_mapping" in truncated:
        offsets = truncated["offset_mapping"]
        spans = [pair for pair in offsets if pair[1] > pair[0]]
        if spans:
            cut_char_idx = spans[-1][1]
    else:
        decoded = tokenizer.decode(
            truncated["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        pos = text.find(decoded)
        if pos >= 0:
            cut_char_idx = pos + len(decoded)
        else:
            cut_char_idx = len(decoded)

    return text[:cut_char_idx], total_tokens, used_tokens


def print_log_snapshot(start_row, end_row, last_truncation_log, last_spelling_log, is_final=False):
    """Tampilkan ringkasan log untuk rentang data tertentu."""
    label = "LOG SNAPSHOT AKHIR" if is_final else "LOG SNAPSHOT"
    print("\n" + "=" * 30)
    print(f"{label} DATA {start_row}-{end_row}")

    if last_truncation_log:
        print(
            f"[TRUNCATE] Baris {last_truncation_log['row_index']}: "
            f"{last_truncation_log['total_tokens']} -> {last_truncation_log['used_tokens']} token "
            f"(limit {last_truncation_log['token_limit']}, "
            f"hilang {last_truncation_log['dropped_tokens']} token)"
        )
        print(f"   - Akhir teks terpakai   : {last_truncation_log['tail_used']}")
        print(f"   - Akhir teks seharusnya : {last_truncation_log['tail_full']}")
        print(f"   - Karakter terbuang     : {last_truncation_log['dropped_chars']}")
    else:
        print("[TRUNCATE] Tidak ada data yang terpotong pada rentang ini.")

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
    client.create_payload_index(
        collection_name=settings.COLLECTION_NAME,
        field_name="rta_id",
        field_schema="integer"
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

    df['total_chars'] = (df['Agreement'].astype(str) + " - " + 
                         df['chapter_name'].astype(str) + " - " + 
                         df['article_name'].astype(str) + ": " + 
                         df['article_text'].astype(str)).str.len()
    
    print("\n=== Statistik Panjang Karakter ===")
    print(df['total_chars'].describe())
    over_limit = len(df[df['total_chars'] > 3000])
    print(f"⚠️ Data di atas 3000 karakter: {over_limit} dari {len(df)} baris")
    print("==================================\n")
    return df

def process_and_upload(client, model, df):
    """Melakukan embedding dan upload data ke Qdrant."""
    points = []
    batch_size = 100
    log_every = 500
    tokenizer = get_model_tokenizer(model)
    embedding_token_limit = model.get_max_seq_length() or getattr(model, "max_seq_length", None) or 384

    print("🚀 Memulai proses Embedding...")
    print(f"⚙️ Limit token embedding (sesuai model): {embedding_token_limit}")
    start_time = time.time()
    changes_count = 0
    truncation_count = 0
    processed_count = 0
    current_window_start = 1
    last_truncation_log = None
    last_spelling_log = None
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

        text_for_embedding = localspelling.convert_spelling(text_to_vector, "us")
        
        if text_to_vector != text_for_embedding:
            changes_count += 1
            orig_words = text_to_vector.split()
            conv_words = text_for_embedding.split()
            spelling_changes = []
            for i in range(min(len(orig_words), len(conv_words))):
                if orig_words[i] != conv_words[i]:
                    spelling_changes.append((orig_words[i], conv_words[i]))

            if spelling_changes:
                shown_changes = spelling_changes[:8]
                last_spelling_log = {
                    "row_index": index,
                    "changes": shown_changes,
                    "extra_changes": len(spelling_changes) - len(shown_changes),
                }

        # Encode (truncation mengikuti batas token model, bukan karakter)
        text_for_encoding, total_tokens, used_tokens = truncate_text_by_token_limit(
            text_for_embedding,
            tokenizer,
            embedding_token_limit,
        )

        if total_tokens > embedding_token_limit:
            truncation_count += 1
            dropped_chars = len(text_for_embedding) - len(text_for_encoding)
            dropped_tokens = total_tokens - used_tokens
            last_truncation_log = {
                "row_index": index,
                "total_tokens": total_tokens,
                "used_tokens": used_tokens,
                "token_limit": embedding_token_limit,
                "dropped_tokens": dropped_tokens,
                "dropped_chars": dropped_chars,
                "tail_used": tail_preview(text_for_encoding),
                "tail_full": tail_preview(text_for_embedding),
            }

        vector = model.encode(text_for_encoding, normalize_embeddings=True).tolist()

        payload = {
            "rta_id": int(row['rta_id']),
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
        processed_count += 1

        if len(points) >= batch_size:
            client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points
            )
            print(f"   ✅ Berhasil upload {len(points)} data (Total terproses: {processed_count})")
            points = []

        if processed_count % log_every == 0:
            print_log_snapshot(
                start_row=current_window_start,
                end_row=processed_count,
                last_truncation_log=last_truncation_log,
                last_spelling_log=last_spelling_log,
            )
            current_window_start = processed_count + 1
            last_truncation_log = None
            last_spelling_log = None
    
    if points:
        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        print(f"   ✅ Berhasil upload sisa {len(points)} data.")

    if processed_count % log_every != 0:
        print_log_snapshot(
            start_row=current_window_start,
            end_row=processed_count,
            last_truncation_log=last_truncation_log,
            last_spelling_log=last_spelling_log,
            is_final=True,
        )
    
    duration = time.time() - start_time
    print("\n" + "="*30)
    print(f"✨ PROSES SELESAI!")
    print(f"📊 Total baris yang dikonversi ke US: {changes_count} dari {len(df)} data.")
    print(f"Total baris yang terkena truncation: {truncation_count} dari {len(df)} data.")
    print(f"⏱️ Waktu total: {duration:.2f} detik.")
    print("="*30 + "\n")

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
