from fastapi import APIRouter
from src.search_models import SearchRequest, SearchResponse, SearchData
from src.search_engine import search_engine

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
        code=200,
        message=msg,
        data=SearchData(          
            count=len(results),   
            rows=results          
        )
    )
