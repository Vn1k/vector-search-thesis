# vector-search-thesis

```
project_root/
├── .env                    # Variabel lingkungan (API Key, Qdrant Host)
├── .gitignore
├── requirements.txt
├── README.md
├── assets/                 # Dataset
├── documentation/          # Dokumen skripsi/teknis
└── src/
    ├── __init__.py
    ├── main.py             # Entry point
    ├── core/               # Konfigurasi utama
    │   ├── __init__.py
    │   └── config.py       # Load env vars (QDRANT_HOST, dll)
    ├── models/             # Pydantic Schemas (Format Request/Response)
    │   ├── __init__.py
    │   └── search.py       # Definisi bentuk JSON input/output
    ├── services/           
    │   ├── __init__.py
    │   └── search_engine.py # Pindahan dari poc.py (Logic Qdrant + AI)
    └── api/                # Router/Endpoint API
        ├── __init__.py
        └── v1/
            ├── __init__.py
            └── endpoints/
                ├── __init__.py
                └── search.py # Endpoint @app.get("/search")
```

to run `uvicorn src.main:app --reload`

to init `python -m src.scripts.ingest`