import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings:
    PROJECT_NAME: str = "Vector Search Thesis"
    API_V1: str = "/api/v1"

    #qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT"))
    COLLECTION_NAME: str = "document_rtas"
    COLLECTION_NAME_RECURSIVE: str = "document_rtas_recursive"
    COLLECTION_NAME_WITHOUT_CHUNKING = "no_chunk_document"
    VECTOR_SIZE: int = 768

    #EVALUATION COLLECTION
    # M = 8 Configurations
    COLLECTION_HYBRID_M8_EF100: str = "rtas_hybrid_m8_ef100"
    COLLECTION_HYBRID_M8_EF200: str = "rtas_hybrid_m8_ef200"
    COLLECTION_HYBRID_M8_EF400: str = "rtas_hybrid_m8_ef400"

    COLLECTION_RECURSIVE_M8_EF400: str = "rtas_recursive_m8_ef400"

    # M = 16 Configurations
    COLLECTION_HYBRID_M16_EF100: str = "rtas_hybrid_m16_ef100"  #default
    COLLECTION_HYBRID_M16_EF200: str = "rtas_hybrid_m16_ef200"
    COLLECTION_HYBRID_M16_EF400: str = "rtas_hybrid_m16_ef400"

    # M = 32 Configurations
    COLLECTION_HYBRID_M32_EF100: str = "rtas_hybrid_m32_ef100"
    COLLECTION_HYBRID_M32_EF200: str = "rtas_hybrid_m32_ef200"
    COLLECTION_HYBRID_M32_EF400: str = "rtas_hybrid_m32_ef400"

    #model
    VECTOR_MODEL: str = "all-mpnet-base-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    DATA_LIMIT = None
    FILE_COVER = str(BASE_DIR / "assets" / "dss_trade_agreements_202602201509.csv")
    FILE_DETAIL = str(BASE_DIR / "assets" / "dss_trade_agreement_detail_formatted_202602201508.csv")
    FILE_CHUNKS = str(BASE_DIR / "assets" / "hybrid_chunks_result_2.csv")
    FILE_CHUNKS_RECURSIVE = str(BASE_DIR / "assets" / "recursive_chunks_result2.csv")

settings = Settings()
