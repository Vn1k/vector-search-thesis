from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    threshold: float = 0.10

class DocumentResult(BaseModel):
    id: int
    agreement: str
    title: str
    content: str
    score: float
    confidence: str

class SearchResponse(BaseModel):
    query: str
    total_found: int
    results: List[DocumentResult]
    message: Optional[str] = None