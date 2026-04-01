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
