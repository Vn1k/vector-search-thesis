"""
Integration module for fine-tuned negation parser.
Replace LLM-based parser with fast model inference.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetuning_negation.inference import NegationParser

# Global parser instance (lazy loading)
_parser = None


def get_parser(model_path="./finetuning_negation/negation_model"):
    """Get or create parser instance."""
    global _parser
    if _parser is None:
        # Try to load fine-tuned model
        if os.path.exists(model_path):
            print(f"[NegationParser] Loading fine-tuned model from {model_path}")
            _parser = NegationParser(model_path)
        else:
            print(
                f"[NegationParser] Model not found at {model_path}, fine-tuned parser unavailable"
            )
            return None
    return _parser


def parse_query_fast(query: str, confidence_threshold: float = 0.7) -> dict:
    """
    Fast query parsing using fine-tuned model.
    Falls back to original query if confidence too low.

    Returns:
        {
            "positive_query": str,
            "excluded_terms": list,
            "iso_code": None,  # Still need separate detection
            "confidence": float,
            "used_finetuned": bool,
        }
    """
    parser = get_parser()

    if parser is None:
        # Fallback: no parsing
        return {
            "positive_query": query,
            "excluded_terms": [],
            "iso_code": None,
            "confidence": 0.0,
            "used_finetuned": False,
        }

    # Parse with fine-tuned model
    result = parser.parse(query, confidence_threshold)

    return {
        "positive_query": result["positive_query"],
        "excluded_terms": result["excluded_terms"],
        "iso_code": None,  # Country detection handled separately
        "confidence": result["confidence"],
        "used_finetuned": True,
    }


def parse_with_fallback(
    query: str, llm_parser_func=None, confidence_threshold: float = 0.7
) -> dict:
    """
    Parse with fine-tuned model, fallback to LLM if confidence low.

    Args:
        query: User query
        llm_parser_func: Original LLM parser function (optional)
        confidence_threshold: Min confidence to accept fine-tuned result

    Returns:
        Parsed result dict
    """
    # Try fine-tuned model first
    fast_result = parse_query_fast(query, confidence_threshold)

    if (
        fast_result["used_finetuned"]
        and fast_result["confidence"] >= confidence_threshold
    ):
        print(
            f"[Parse] Using fine-tuned model (confidence: {fast_result['confidence']:.2f})"
        )
        return fast_result

    # Fallback to LLM if available
    if llm_parser_func:
        print(
            f"[Parse] Fine-tuned confidence low ({fast_result['confidence']:.2f}), falling back to LLM"
        )
        return llm_parser_func(query)

    # Return low-confidence fine-tuned result
    return fast_result


# For testing
if __name__ == "__main__":
    test_queries = [
        "tariffs on textiles excluding silk and cotton",
        "agricultural products excluding rice",
        "rules of origin without cumulation",
        "customs procedures",
    ]

    print("Testing fine-tuned parser integration...\n")

    for query in test_queries:
        result = parse_query_fast(query)
        print(f"Query: {query}")
        print(f"  Positive: {result['positive_query']}")
        print(f"  Excluded: {result['excluded_terms']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Used fine-tuned: {result['used_finetuned']}")
        print()
