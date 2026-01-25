from fastapi import FastAPI
from src.core.config import settings
from src.api.v1.endpoints import search

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(search.router, prefix=settings.API_V1, tags=["Search"])

@app.get("/")
def root():
    return {"message": "Vector Search Engine API is Running 🚀"}