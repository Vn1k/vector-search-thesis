# vector-search-thesis

```
project_root/
├── .env                    # Variabel lingkungan (API Key, Qdrant Host)
├── .gitignore
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── main.py             # Entry point
    ├── config.py           # Load env vars (QDRANT_HOST, dll)
    ├── search_models.py    # Pydantic Schemas (Format Request/Response)
    ├── search_engine.py    # Logic Qdrant + AI
    ├── search_endpoint.py  # Router/Endpoint API
    ├── ingest.py           # Script ingest data
    ├── assets/             # Dataset
    └── documentation/      # Dokumen skripsi/teknis
```

to run `uvicorn src.main:app --reload`

to init `python -m src.ingest`

## Export kandidat ground truth

Untuk mengambil 10 agreement teratas dari search dan menyimpan top 3 artikel dari
setiap agreement ke JSON:

```bash
python -m src.export_ground_truth_candidates "rules of origin" \
  -o evaluation/ground_truth_candidates_rules_of_origin.json
```

Secara default script memakai endpoint yang sama dengan `src/search.sh`, yaitu
`http://localhost:8000/api/v1/search`. Pastikan backend sudah berjalan dengan
`uvicorn src.main:app --reload`.

Jika ingin memanggil fungsi search langsung tanpa endpoint:

```bash
python -m src.export_ground_truth_candidates "rules of origin" \
  --source function \
  -o evaluation/ground_truth_candidates_rules_of_origin.json
```
