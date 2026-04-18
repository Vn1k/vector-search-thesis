import math
import json
import logging
import pycountry
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer
from spellchecker import SpellChecker
import localspelling
from ollama import chat
from src.config import settings

# Import fine-tuned negation parser (optional)
try:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from finetuning_negation.FLAN.inference import NegationParser

    FINETUNED_AVAILABLE = True
except ImportError:
    FINETUNED_AVAILABLE = False

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        use_reranker: bool = False,
        use_llm: bool = False,
        use_finetuned: bool = True,
    ):
        logger.info("[Init] Initializing model & database")
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.vector_model = SentenceTransformer(settings.VECTOR_MODEL)
        self.spell = SpellChecker()

        # ─── Reranker (opsional) ──────────────────────────────────────────────
        self.use_reranker = use_reranker
        if self.use_reranker:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
            logger.info("[Init] Reranker: ON")
        else:
            self.reranker = None
            logger.info("[Init] Reranker: OFF (vector score only)")

        # ─── Fine-tuned Query Parser (priority) ────────────────────────────────
        self.use_finetuned = use_finetuned and FINETUNED_AVAILABLE
        self.finetuned_parser = None
        if self.use_finetuned:
            try:
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "finetuning_negation",
                    "FLAN",                  
                    "negation_model_t5_v2",
                )
                self.finetuned_parser = NegationParser(model_path)
                logger.info("[Init] Fine-tuned Parser: ON")
            except Exception as e:
                logger.warning(f"[Init] Fine-tuned parser failed to load: {e}")
                self.use_finetuned = False
        else:
            logger.info("[Init] Fine-tuned Parser: OFF")

        # ─── LLM Query Parser (fallback/optional) ─────────────────────────────
        self.use_llm = use_llm
        self.LLM_MODEL = "nemotron-3-nano:4b"
        if self.use_llm:
            logger.info(f"[Init] LLM Query Parser: ON ({self.LLM_MODEL})")
        else:
            logger.info("[Init] LLM Query Parser: OFF")
        # ──────────────────────────────────────────────────────────────────────

        logger.info("[Init] System ready")

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
        logger.info(f"[LLM Parser] Original query: '{user_query}'")

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

            logger.info(f"[LLM Parser] positive_query: '{result['positive_query']}'")
            logger.info(f"[LLM Parser] excluded_terms: {result['excluded_terms']}")
            logger.info(f"[LLM Parser] iso_code: {result['iso_code']}")

            return result

        except Exception as e:
            logger.warning(
                f"[LLM Parser] Parsing failed: {e} — fallback ke query original"
            )
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
        logger.info(f"[Search] === New Search Request ===")
        logger.info(f"[Search] Original user query: '{query}'")
        logger.info(f"[Search] Parameters: limit={limit}, threshold={threshold}")

        # ─── Step 1: Query parsing (fine-tuned atau LLM) ──────────────────────
        excluded_terms = []
        llm_iso = None

        if self.use_finetuned and self.finetuned_parser:
            # Use fine-tuned model for fast parsing
            logger.info("[QueryParser] Using fine-tuned model")
            parsed = self.finetuned_parser.parse(query)
            search_query = parsed["positive_query"]
            excluded_terms = parsed["excluded_terms"]
            # ISO code will be detected in Step 2
            logger.info(f"[Finetuned] positive_query: '{search_query}'")
            logger.info(f"[Finetuned] excluded_terms: {excluded_terms}")
            logger.info(f"[Finetuned] confidence: {parsed.get('confidence', 0):.3f}")
        elif self.use_llm:
            # Fallback to LLM parsing
            parsed = self.llm_query_parser(query)
            search_query = parsed["positive_query"]
            excluded_terms = parsed["excluded_terms"]
            llm_iso = parsed["iso_code"]
        else:
            # No parsing, use original query
            search_query = query
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 2: Deteksi negara (rule-based, jika tidak dari LLM) ─────────
        detected_iso = llm_iso or self.detect_iso_from_query(search_query)

        qdrant_filter = None
        if detected_iso:
            logger.info(
                f"[QueryTransform] Detected country: {detected_iso}, using filter"
            )
            qdrant_filter = Filter(
                must=[FieldCondition(key="iso", match=MatchValue(value=detected_iso))]
            )
        else:
            logger.info("[QueryTransform] No detected country, using global search")
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 3: Fix tokenization (fine-tuned) + dialect normalization ────
          # Original spell checker for non-finetuned mode
        if not self.use_finetuned:
            # Spell checker hanya untuk non-finetuned mode
            words = search_query.split()
            misspelled = self.spell.unknown(words)
            corrected = []
            for word in words:
                if word in misspelled:
                    correction = self.spell.correction(word)
                    corrected.append(correction if correction else word)
                else:
                    corrected.append(word)
            search_query = " ".join(corrected)

        # Apply UK/US spelling normalization
        spelling_query = localspelling.convert_spelling(search_query, "us")
        excluded_terms = [
            localspelling.convert_spelling(term, "us") for term in excluded_terms
        ]

        if search_query != spelling_query:
            logger.info(
                f"[QueryTransform] Dialect normalization: '{search_query}' -> '{spelling_query}'"
            )
        else:
            logger.debug(f"[QueryTransform] Dialect normalization: no changes")

        final_query = spelling_query
        logger.info(f"[QueryTransform] Final query to Qdrant: '{final_query}'")
        # ──────────────────────────────────────────────────────────────────────

        # ─── Step 4: Vector search ────────────────────────────────────────────
        query_vector = self.vector_model.encode(final_query).tolist()
        logger.debug(
            f"[QueryTransform] Query vector generated (dim: {len(query_vector)})"
        )

        hits = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=20,
            with_payload=True,
        ).points

        logger.info(f"[QueryTransform] Qdrant returned {len(hits)} results")

        if not hits:
            logger.info("[QueryTransform] No results from Qdrant, returning empty")
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
            logger.info(
                f"[NegationFilter] Starting exclusion filter with terms: {excluded_terms}"
            )
            logger.info(f"[NegationFilter] Results before filter: {len(results)}")

            def find_excluded_terms(item: dict) -> list:
                """Return list of excluded terms found in the document."""
                text = item.get("article_text", "").lower()
                return [term for term in excluded_terms if term.lower() in text]

            filtered_results = []
            excluded_docs_log = []

            for item in results:
                found_terms = find_excluded_terms(item)
                if found_terms:
                    # Log excluded document details
                    doc_info = f"{item.get('agreement', '-')} - {item.get('chapter_name', '-')} - {item.get('article_name', '-')}"
                    excluded_docs_log.append({"doc": doc_info, "terms": found_terms})
                    logger.info(
                        f"[NegationFilter] EXCLUDED: '{doc_info}' matched terms: {found_terms}"
                    )
                else:
                    filtered_results.append(item)

            results = filtered_results
            after = len(results)

            logger.info(
                f"[NegationFilter] Results after filter: {after} (removed {len(results) - after + len(excluded_docs_log)} documents)"
            )
            logger.info(f"[NegationFilter] Excluded terms applied: {excluded_terms}")
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
            grouped[rta_id]["articles"].append(
                {
                    "id": item["id"],
                    "chapter_name": item["chapter_name"],
                    "article_name": item["article_name"],
                    "article_text": item["article_text"],
                    "article_text_display": item["article_text_display"],
                    "score": item["score"],
                    "confidence": item["confidence"],
                }
            )

        grouped_list = sorted(
            grouped.values(), key=lambda x: x["best_score"], reverse=True
        )
        final_results = grouped_list[:limit]
        logger.info(f"[Search] Returning {len(final_results)} grouped results")
        logger.info(f"[Search] === Search Request Complete ===")
        return final_results
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
    use_llm=False,  # Using fine-tuned parser instead
    use_finetuned=True,
)
