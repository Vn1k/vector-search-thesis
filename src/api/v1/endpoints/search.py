from fastapi import APIRouter
from src.models.search import SearchRequest, SearchResponse
from src.services.search_engine import search_engine

router = APIRouter()

@router.post('/search', response_model=SearchResponse)
async def search_documents (request: SearchRequest):
    results = search_engine.search(
        query=request.query,
        limit=request.limit,
        threshold=request.threshold
    )

    msg = "Success" if results else "No relevan document found"

    return SearchResponse(
        query=request.query,
        total_found=len(results),
        results=results,
        message=msg
    )