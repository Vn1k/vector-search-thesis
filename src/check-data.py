import sys
import pandas as pd
from src.config import settings

def analyze_character_length():
    """Membaca dan membersihkan data CSV, lalu menampilkan statistik karakter."""
    print(f"📂 Membaca file CSV dari {settings.FILE_DETAIL} dan {settings.FILE_COVER}...")
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
    print(f"Total data sebelum hapus duplikat: {len(df)}")

    # Hapus duplikat
    df = df.drop_duplicates(subset=['article_text', 'Agreement'])
    print(f"Total data setelah hapus duplikat: {len(df)}")
    
    # Batasi jumlah data jika diperlukan (diatur di settings)
    if hasattr(settings, 'DATA_LIMIT') and settings.DATA_LIMIT:
        df = df.head(settings.DATA_LIMIT)
        
    print(f"📊 Total Data Siap Dianalisis: {len(df)} baris.\n")

    # Menggabungkan teks untuk dihitung panjang karakternya
    df['total_chars'] = (df['Agreement'].astype(str) + " - " + 
                         df['chapter_name'].astype(str) + " - " + 
                         df['article_name'].astype(str) + ": " + 
                         df['article_text'].astype(str)).str.len()
    
    # Menampilkan Statistik
    print("\n=== Statistik Panjang Karakter ===")
    
    # Menampilkan rata-rata, min, max, dll.
    print(df['total_chars'].describe())
    
    # Menghitung berapa banyak data yang ukurannya terlalu panjang (misal: > 3000 karakter)
    batas_karakter = 3000
    over_limit = len(df[df['total_chars'] > batas_karakter])
    print(f"\n⚠️ Data di atas {batas_karakter} karakter: {over_limit} dari {len(df)} baris")
    print("==================================\n")
    
    return df

if __name__ == "__main__":
    # Jalankan fungsi analisis
    df_result = analyze_character_length()