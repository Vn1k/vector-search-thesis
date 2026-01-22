# CATATAN

Sistem saat ini menggunakan bi encoder lewat qdrant, dimana cara kerja bi encoder itu mengubah kalimat menjadi angka dan mengukur jaraknya.

Untuk mengubah hasil reranker dalam bentuk persen akan menggunakan rumus sigmoid


# 📝 CATATAN TEKNIS SYSTEM (Developer Notes)

### 1. Arsitektur Sistem (Two-Stage Retrieval)

Sistem menggunakan pendekatan **Hybrid / Two-Stage** untuk menyeimbangkan kecepatan dan akurasi:

* **Stage 1 (Retrieval):** Menggunakan **Bi-Encoder** (via Qdrant).
* *Tujuan:* **Recall Tinggi**. Mengambil kandidat kasar (Top 20) dari ribuan dokumen secepat kilat.
* *Sifat:* Cepat, tapi kadang kurang peka nuansa (hanya melihat kemiripan topik).


* **Stage 2 (Re-ranking):** Menggunakan **Cross-Encoder** (via Sentence-Transformers).
* *Tujuan:* **Precision Tinggi**. Menilai ulang 20 kandidat untuk menentukan urutan 1-5 terbaik.
* *Sifat:* Lambat (komputasi berat), tapi sangat cerdas memahami intensi pertanyaan.

### 2. Spesifikasi Model

* **Vector Model (Bi-Encoder):** `all-mpnet-base-v2`
* **Dimensi:** **768** (Wajib set ini di config Qdrant).
* **Max Token:** 384 token.
* **Alasan:** Lebih cerdas dari `MiniLM` (384 dim), tapi masih cukup ringan untuk laptop.


* **Reranker Model (Cross-Encoder):** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Input:** Pasangan `(Query, Dokumen)`.
* **Output:** *Logits* (Raw Score), bukan 0-1.

### 3. Strategi Data (Contextual Embedding)

* **Teknik:** Menggabungkan metadata hierarkis ke dalam teks sebelum di-vektorisasi.
* **Format String:** `f"{Agreement} - {Bab} - {Judul}: {Isi}"`
* **Tujuan:** Agar vektor tidak "buta konteks". Vektor harus tahu bahwa pasal ini milik perjanjian "Indonesia" atau "Vietnam".
* **Constraint:** Teks dipotong (`[:1000]`) sebelum masuk model untuk mencegah *truncation* (pemotongan paksa) pada informasi penting di akhir kalimat karena batas token model.

### 4. Mekanisme Filtering (ISO)

* **Jenis:** **Pre-filtering** (Filter sebelum search).
* **Teknologi:** Qdrant Payload Indexing.
* **Logic:** Jika ML mendeteksi negara (misal: "IDN"), sistem memaksa Qdrant hanya melihat dokumen dengan `iso: IDN`. Ini mengeliminasi 100% *hallucination* geografis.

### 5. Interpretasi Scoring (Math)

* **Vector Score (Qdrant):**
* Menggunakan **Cosine Similarity**.
* Range: `-1` s/d `1`. (Biasanya 0.4 - 0.8 untuk teks).
* Semakin mendekati 1 semakin mirip.


* **Reranker Score (Cross-Encoder):**
* Menggunakan **Logits** (Unbounded).
* Range: Bisa `-negatif` sampai `+positif`.
* Rule of thumb: `> 3.0` (Sangat Akurat), `0-3` (Relevan), `< 0` (Sampah).
* **Konversi UI:** Gunakan **Sigmoid Function** untuk mengubah Logits jadi Persentase (0-100%).
* 

### 6. Evaluasi Metrik

* **Ground Truth:** ID dokumen yang "seharusnya" muncul.
* **MRR (Mean Reciprocal Rank):** Mengukur seberapa tinggi ranking jawaban benar. (Ranking 1 nilai 1.0, Ranking 2 nilai 0.5).
* **Hit Rate @ K:** Persentase keberhasilan jawaban benar muncul di dalam Top-K (Top 3 atau Top 5).