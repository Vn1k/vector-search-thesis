from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.0

# Model untuk artikel di dalam group
class ArticleResult(BaseModel):
    id: int
    chapter_name: str
    article_id  : str
    article_number: str
    article_name: str
    article_text: str
    article_text_display: str
    score: float
    confidence: str

# Model untuk tiap group (per agreement)
class GroupedResult(BaseModel):
    rta_id: int
    agreement: str
    acronym: str
    iso: List[str]
    best_score: float
    best_confidence: str
    articles: List[ArticleResult] 

class SearchData(BaseModel):
    count: int
    rows: List[GroupedResult]

class SearchResponse(BaseModel):
    code: int
    message: Optional[str] = None
    data: SearchData