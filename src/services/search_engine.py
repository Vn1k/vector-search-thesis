from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.core.config import settings
import math
import pycountry

class SearchService:
    def __init__(self):
        print("Initialization model & database")
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.vector_model = SentenceTransformer(settings.VECTOR_MODEL)
        self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        print("System ready")
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def detect_iso_from_query(self, query):
        query_lower = query.lower()

        manual_map = {
            "russia": "RUS",           
            "vietnam": "VNM",          
            "usa": "USA",              
            "united states": "USA",
            "uk": "GBR",               
            "england": "GBR",
            "south korea": "KOR",      
            "korea": "KOR",
            "brunei": "BRN",           
            "laos": "LAO",
        }

        for name, code in manual_map.items():
            if name in query_lower:
                return code

        for country in pycountry.countries:
            country_name = country.name.lower()

            if country_name in query_lower:
                return country.alpha_3
            
        return None

    def search(self, query: str, limit: int = 5, threshold: float = 0.1):
        detected_iso = self.detect_iso_from_query(query)
        qdrant_filter = None
        if detected_iso:
            print(f"ML detect country: {detected_iso}, using filter")

            qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="iso",
                            match=MatchValue(value=detected_iso)
                        )
                    ]
                )
        else:
            print("No detected country, using global")

        query_vector = self.vector_model.encode(query).tolist()

        #Retrieval tahap 1 (vector search happens here)
        hits = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,  
            query_filter=qdrant_filter,
            limit=20,
            with_payload=True
        ).points

        if not hits:
            return []
        
        # Re-ranking
        cross_inp = []
        for hit in hits:
            pl = hit.payload
            context_text = f"{pl['Agreement']} - {pl['chapter_name']} - {pl['article_name']}: {pl['article_text']}"
            cross_inp.append([query, context_text])
        
        cross_scores = self.reranker.predict(cross_inp)

        results = []
        for idx, hit in enumerate(hits):
            raw_score = cross_scores[idx]
            score_percent = self.sigmoid(raw_score)

            if score_percent < threshold:
                continue

            results.append({
                "id": hit.id,
                "agreement": hit.payload.get('Agreement', '-'),
                "chapter_name": hit.payload.get('chapter_name', '-'),
                "article_name": hit.payload.get('article_name', '-'),
                "article_text": hit.payload.get('article_text', '-'),
                "article_text_display": hit.payload.get('article_text_display', '-'),
                "iso": hit.payload.get('iso', '-'),
                "score": score_percent,
                "confidence": f"{score_percent:.4%}"
            })

        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results[:limit]

search_engine = SearchService()
        
