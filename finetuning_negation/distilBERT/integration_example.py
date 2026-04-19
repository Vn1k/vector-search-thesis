"""
Example modification to search_engine.py to use fine-tuned parser.

This shows what changes to make in your existing search_engine.py
"""

# ============================================================================
# OPTION 1: Replace LLM parser completely (if fine-tuned model is good enough)
# ============================================================================

# In search_engine.py, modify llm_query_parser method:


def llm_query_parser_finetuned(self, user_query: str) -> dict:
    """
    Fast query parsing using fine-tuned model instead of LLM.
    """
    from finetuning_negation.integration import parse_query_fast

    result = parse_query_fast(user_query)

    # Add iso_code detection (not handled by fine-tuned model)
    result["iso_code"] = self.detect_iso_from_query(result["positive_query"])

    logger.info(f"[Finetuned Parser] positive_query: '{result['positive_query']}'")
    logger.info(f"[Finetuned Parser] excluded_terms: {result['excluded_terms']}")
    logger.info(f"[Finetuned Parser] confidence: {result['confidence']:.2f}")

    return result


# ============================================================================
# OPTION 2: Hybrid approach (fine-tuned + LLM fallback)
# ============================================================================


def llm_query_parser_hybrid(self, user_query: str) -> dict:
    """
    Try fine-tuned model first, fallback to LLM if confidence low.
    """
    from finetuning_negation.integration import parse_query_fast

    # Try fine-tuned model
    fast_result = parse_query_fast(user_query, confidence_threshold=0.7)

    if fast_result["used_finetuned"] and fast_result["confidence"] >= 0.7:
        # Fine-tuned model confident enough
        fast_result["iso_code"] = self.detect_iso_from_query(
            fast_result["positive_query"]
        )

        logger.info(
            f"[Hybrid Parser] Using fine-tuned model (confidence: {fast_result['confidence']:.2f})"
        )
        logger.info(
            f"[Hybrid Parser] positive_query: '{fast_result['positive_query']}'"
        )
        logger.info(f"[Hybrid Parser] excluded_terms: {fast_result['excluded_terms']}")

        return fast_result

    # Fallback to LLM
    logger.info(
        f"[Hybrid Parser] Fine-tuned confidence low ({fast_result['confidence']:.2f}), using LLM"
    )
    return self.llm_query_parser(user_query)  # Original LLM method


# ============================================================================
# OPTION 3: A/B Testing (compare both approaches)
# ============================================================================


def llm_query_parser_with_comparison(self, user_query: str) -> dict:
    """
    Parse with both methods and log comparison for evaluation.
    """
    import time
    from finetuning_negation.integration import parse_query_fast

    # Fine-tuned model
    start = time.time()
    fast_result = parse_query_fast(user_query)
    fast_time = time.time() - start

    # LLM
    start = time.time()
    llm_result = self.llm_query_parser(user_query)  # Original method
    llm_time = time.time() - start

    # Log comparison
    logger.info(f"[Comparison] Query: '{user_query}'")
    logger.info(
        f"[Comparison] Fine-tuned: {fast_time * 1000:.1f}ms, pos='{fast_result['positive_query']}', ex={fast_result['excluded_terms']}"
    )
    logger.info(
        f"[Comparison] LLM: {llm_time * 1000:.1f}ms, pos='{llm_result['positive_query']}', ex={llm_result['excluded_terms']}"
    )
    logger.info(f"[Comparison] Speedup: {llm_time / fast_time:.1f}x")

    # Return LLM result (ground truth) for now
    return llm_result


# ============================================================================
# How to integrate into SearchService class
# ============================================================================

# In search_engine.py, add to __init__:
# self.parser_mode = "finetuned"  # or "hybrid" or "llm"

# Then in search() method:
# if self.parser_mode == "finetuned":
#     parsed = self.llm_query_parser_finetuned(query)
# elif self.parser_mode == "hybrid":
#     parsed = self.llm_query_parser_hybrid(query)
# else:
#     parsed = self.llm_query_parser(query)  # Original
