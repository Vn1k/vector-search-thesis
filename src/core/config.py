import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    PROJECT_NAME: str = "Vector Search Thesis"
    API_V1: str = "/api/v1"

    #qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT"))
    COLLECTION_NAME: str = "document_rtas"
    VECTOR_SIZE: int = 768

    #model
    VECTOR_MODEL: str = "all-mpnet-base-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    DATA_LIMIT = None
    FILE_COVER = str(BASE_DIR / "assets" / "dss_trade_agreements_202511042008.csv")
    FILE_DETAIL = str(BASE_DIR / "assets" / "dss_trade_agreement_detail_formatted_202511042008.csv")

settings = Settings()