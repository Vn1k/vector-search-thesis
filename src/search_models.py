from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    threshold: float = 0.10

class DocumentResult(BaseModel):
    id: int
    agreement: str
    chapter_name: str
    article_name: str
    article_text: str
    article_text_display: str
    iso: list[str]
    score: float
    confidence: str

class SearchData(BaseModel):
    count: int
    rows: List[DocumentResult]

class SearchResponse(BaseModel):
    code: int
    message: Optional[str] = None
    data: SearchData