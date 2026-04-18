"""
inference.py
============
Inference module untuk fine-tuned T5 negation parser.
Dipakai oleh search_service.py via NegationParser class.

Usage standalone:
    python inference.py
    python inference.py --model_dir ./FLAN/negation_model_t5_v2
"""

import argparse
import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ─────────────────────────── CONFIG ────────────────────────────────────────
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "negation_model_t5_v2")
MAX_INPUT_LENGTH    = 128
MAX_OUTPUT_LENGTH   = 128
PROMPT_PREFIX       = (
    "Parse the trade agreement search query. "
    "Extract the positive topic and any excluded terms. "
    "Query: "
)
# ───────────────────────────────────────────────────────────────────────────


class NegationParser:
    """
    Fine-tuned T5 parser untuk mengekstrak:
    - positive_query  : topik utama yang dicari
    - excluded_terms  : list term yang ingin di-exclude
    - iso_code        : kode negara ISO 3 huruf (atau None)
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_DIR):
        self.device = self._resolve_device()
        logger.info(f"[NegationParser] Loading model from: {model_path}")
        logger.info(f"[NegationParser] Device: {self.device}")

        self.tokenizer = self._load_tokenizer(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        logger.info("[NegationParser] Model ready")

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _load_tokenizer(model_path: str):
        try:
            return AutoTokenizer.from_pretrained(model_path)
        except AttributeError as exc:
            if "has no attribute 'keys'" not in str(exc):
                raise

            logger.warning(
                "[NegationParser] Tokenizer config incompatible with current "
                "Transformers version. Retrying with extra_special_tokens={}."
            )
            return AutoTokenizer.from_pretrained(
                model_path,
                extra_special_tokens={},
            )

    def parse(self, query: str) -> dict:
        """
        Parse user query menjadi structured dict.

        Returns:
            {
                'positive_query' : str,
                'excluded_terms' : list[str],
                'iso_code'       : str | None,
                'confidence'     : float,   # selalu 1.0 (T5 tidak output score)
                'raw'            : str,     # raw output model
            }
        """
        prompt = PROMPT_PREFIX + query

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = self._parse_output(raw)
        result["confidence"] = 1.0
        result["raw"] = raw

        logger.info(f"[NegationParser] Query   : '{query}'")
        logger.info(f"[NegationParser] Raw out : '{raw}'")
        logger.info(f"[NegationParser] Parsed  : {result}")

        return result

    def parse_batch(self, queries: list[str]) -> list[dict]:
        """
        Parse banyak query sekaligus (lebih efisien untuk batch processing).
        """
        prompts = [PROMPT_PREFIX + q for q in queries]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        results = []
        for i, output in enumerate(outputs):
            raw = self.tokenizer.decode(output, skip_special_tokens=True)
            result = self._parse_output(raw)
            result["confidence"] = 1.0
            result["raw"] = raw
            results.append(result)

        return results

    # ─── Internal parser ───────────────────────────────────────────────────
    def _parse_output(self, text: str) -> dict:
        """
        Parse raw model output:
        'positive: X | excluded: Y, Z | country: IDN'
        → dict
        """
        result = {
            "positive_query": text,   # fallback: kembalikan text asli
            "excluded_terms": [],
            "iso_code": None,
        }

        parts = [p.strip() for p in text.split("|")]
        for part in parts:
            if part.startswith("positive:"):
                val = part[len("positive:"):].strip()
                if val:
                    result["positive_query"] = val

            elif part.startswith("excluded:"):
                val = part[len("excluded:"):].strip()
                if val.lower() != "none" and val:
                    result["excluded_terms"] = [
                        t.strip() for t in val.split(",") if t.strip()
                    ]

            elif part.startswith("country:"):
                val = part[len("country:"):].strip()
                result["iso_code"] = None if val.lower() == "none" else val or None

        return result


# ─────────────────────────── STANDALONE TEST ───────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    negation_parser = NegationParser(args.model_dir)

    test_queries = [
        "Trade in Goods excluding Agriculture",
        "Rules of Origin without rice",
        "ASEAN trade remedies excluding anti-dumping and subsidies",
        "Japan Rules of Origin excluding rice",
        "Trade in Services except financial services",
        "Intellectual Property Rights",
        "Vietnam Sanitary and Phytosanitary Measures aside from automotive",
        "Digital Trade not including Paperless Trade",
    ]

    print("\n" + "=" * 65)
    print("  INFERENCE TEST — Fine-tuned Negation Parser")
    print("=" * 65)

    for query in test_queries:
        result = negation_parser.parse(query)
        print(f"\n  Query    : {query}")
        print(f"  Positive : {result['positive_query']}")
        print(f"  Excluded : {result['excluded_terms']}")
        print(f"  Country  : {result['iso_code']}")

    print("\n" + "=" * 65)

    # Test batch
    print("\n  BATCH TEST (2 queries sekaligus)")
    print("=" * 65)
    batch_results = negation_parser.parse_batch(test_queries[:2])
    for q, r in zip(test_queries[:2], batch_results):
        print(f"\n  Query    : {q}")
        print(f"  Positive : {r['positive_query']}")
        print(f"  Excluded : {r['excluded_terms']}")
        print(f"  Country  : {r['iso_code']}")
    print()


if __name__ == "__main__":
    main()
