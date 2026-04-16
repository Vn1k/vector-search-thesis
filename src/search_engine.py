import math
import json
import pycountry
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer
from spellchecker import SpellChecker
import localspelling
from ollama import chat
from src.config import settings


class SearchService:
    def __init__(self, use_reranker: bool = False, use_llm: bool = False):
        print("Initialization model & database")
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.vector_model = SentenceTransformer(settings.VECTOR_MODEL)
        self.spell = SpellChecker()

        # ─── Reranker (opsional) ──────────────────────────────────────────────
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
            print("Reranker: ON")
        else:
            self.reranker = None
            print("Reranker: OFF (vector score only)")

        # ─── LLM Query Parser (opsional) ──────────────────────────────────────
        self.use_llm = use_llm
        self.LLM_MODEL = "nemotron-3-nano:4b" 
        if self.use_llm:
            print(f"LLM Query Parser: ON ({self.LLM_MODEL})")
        else:
            print("LLM Query Parser: OFF")
        # ──────────────────────────────────────────────────────────────────────

        print("System ready")

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # ─── LLM: Parse query untuk negation handling ─────────────────────────────
    def llm_query_parser(self, user_query: str) -> dict:
        """
        Kirim query ke LLM (Ollama) untuk diekstrak:
        - positive_query  : topik utama (tanpa kata negasi)
        - excluded_terms  : konsep yang TIDAK ingin dilihat user
        - iso_code        : kode negara 3 huruf jika disebutkan

        Return fallback dict jika LLM gagal.
        """
        prompt = f"""
You are a legal document assistant for a trade agreement search engine.
Analyze the following user query and extract structured information.

Rules:
1. positive_query: The main topic to search for. REMOVE all negation phrases (e.g., "not about", "excluding", "without").
2. excluded_terms: A list of specific concepts the user does NOT want (empty list [] if none).
3. iso_code: A 3-letter ISO country code if a country is mentioned (null if none).

Query: "{user_query}"

Respond ONLY in valid JSON. No explanation. No markdown.
Example: {{"positive_query": "textile tariffs", "excluded_terms": ["silk", "cotton"], "iso_code": "VNM"}}
"""

        try:
            response = chat(
                model=self.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.message.content.strip()

            # Bersihkan jika LLM tetap bungkus dengan markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw.strip())

            result = {
                "positive_query": parsed.get("positive_query", user_query),
                "excluded_terms": parsed.get("excluded_terms", []),
                "iso_code": parsed.get("iso_code", None),
            }

            print(f"[LLM] positive_query : {result['positive_query']}")
            print(f"[LLM] excluded_terms : {result['excluded_terms']}")
            print(f"[LLM] iso_code       : {result['iso_code']}")

            return result

        except Exception as e:
            print(f"[LLM] Parsing failed: {e} — fallback ke query original")
            return {
                "positive_query": user_query,
                "excluded_terms": [],
                "iso_code": None,
            }
    # ──────────────────────────────────────────────────────────────────────────

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
            if country.name.lower() in query_lower:
                return country.alpha_3

        return None

    def search(self, query: str, limit: int = 5, threshold: float = 0.1):

        # ─── Step 1: LLM query parsing (jika aktif) ───────────────────────────
        excluded_terms = []

        if self.use_llm:
            parsed = self.llm_query_parser(query)
            search_query = parsed["positive_query"]    # query bersih tanpa negasi
            excluded_terms = parsed["excluded_terms"]  # akan difilter nanti
            llm_iso = parsed["iso_code"]
        else:
            search_query = query
            llm_iso = None
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 2: Deteksi negara (LLM prioritas, fallback rule-based) ──────
        detected_iso = llm_iso or self.detect_iso_from_query(search_query)

        qdrant_filter = None
        if detected_iso:
            print(f"Detected country: {detected_iso}, using filter")
            qdrant_filter = Filter(
                must=[FieldCondition(key="iso", match=MatchValue(value=detected_iso))]
            )
        else:
            print("No detected country, using global")
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 3: Spell correction + dialect normalization ─────────────────
        words = search_query.split()
        misspelled = self.spell.unknown(words)
        corrected = []
        for word in words:
            if word in misspelled:
                correction = self.spell.correction(word)
                corrected.append(correction if correction else word)
            else:
                corrected.append(word)

        typo_query = " ".join(corrected)
        if search_query != typo_query:
            print(f"Typo corrected : '{search_query}' -> '{typo_query}'")

        spelling_query = localspelling.convert_spelling(typo_query, "us")
        if typo_query != spelling_query:
            print(f"Dialect normalized: '{typo_query}' -> '{spelling_query}'")

        final_query = spelling_query
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 4: Vector search ────────────────────────────────────────────
        query_vector = self.vector_model.encode(final_query).tolist()

        hits = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=20,
            with_payload=True,
        ).points

        if not hits:
            return []
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 5: Scoring ──────────────────────────────────────────────────
        if self.use_reranker:
            cross_inp = []
            for hit in hits:
                pl = hit.payload
                context_text = f"{pl['Agreement']} - {pl['chapter_name']} - {pl['article_name']}: {pl['article_text']}"
                cross_inp.append([final_query, context_text])

            cross_scores = self.reranker.predict(cross_inp)

            results = []
            for idx, hit in enumerate(hits):
                score_percent = self.sigmoid(cross_scores[idx])
                if score_percent < threshold:
                    continue
                results.append(self._build_result(hit, score_percent))
        else:
            results = []
            for hit in hits:
                score_percent = float(hit.score)
                if score_percent < threshold:
                    continue
                results.append(self._build_result(hit, score_percent))
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 6: Filter excluded_terms (hasil LLM negation) ───────────────
        if excluded_terms:
            def contains_excluded(item: dict) -> bool:
                text = item.get("article_text", "").lower()
                return any(term.lower() in text for term in excluded_terms)

            before = len(results)
            results = [r for r in results if not contains_excluded(r)]
            after = len(results)
            if before != after:
                print(f"[LLM Filter] Removed {before - after} results → excluded: {excluded_terms}")
        # ──────────────────────────────────────────────────────────────────────

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # ─── Step 7: Grouping per RTA ─────────────────────────────────────────
        grouped = {}
        for item in results:
            rta_id = item["rta_id"]
            if rta_id not in grouped:
                grouped[rta_id] = {
                    "rta_id": rta_id,
                    "agreement": item["agreement"],
                    "acronym": item["acronym"],
                    "iso": item["iso"],
                    "best_score": item["score"],
                    "best_confidence": item["confidence"],
                    "articles": [],
                }
            grouped[rta_id]["articles"].append({
                "id": item["id"],
                "chapter_name": item["chapter_name"],
                "article_name": item["article_name"],
                "article_text": item["article_text"],
                "article_text_display": item["article_text_display"],
                "score": item["score"],
                "confidence": item["confidence"],
            })

        grouped_list = sorted(grouped.values(), key=lambda x: x["best_score"], reverse=True)
        return grouped_list[:limit]
        # ──────────────────────────────────────────────────────────────────────

    def _build_result(self, hit, score_percent: float) -> dict:
        return {
            "id": hit.id,
            "rta_id": hit.payload.get("rta_id"),
            "agreement": hit.payload.get("Agreement", "-"),
            "chapter_name": hit.payload.get("chapter_name", "-"),
            "article_name": hit.payload.get("article_name", "-"),
            "article_text": hit.payload.get("article_text", "-"),
            "article_text_display": hit.payload.get("article_text_display", "-"),
            "acronym": hit.payload.get("acronym", "-"),
            "iso": hit.payload.get("iso", "-"),
            "score": score_percent,
            "confidence": f"{score_percent:.4%}",
        }


search_engine = SearchService(
    use_reranker=False,  
    use_llm=False,      
)
