"""
Evaluate fine-tuned model against LLM (ground truth).
Compare accuracy and speed.
"""

import json
import time
from inference import NegationParser
from ollama import chat


def evaluate_model(
    test_data_path, model_path="./negation_model", llm_model="nemotron-3-nano:4b"
):
    """Evaluate fine-tuned model against LLM baseline."""

    # Load test data
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Initialize parser
    print("Loading fine-tuned model...")
    parser = NegationParser(model_path)

    print(f"\nEvaluating on {len(test_data)} samples...\n")

    results = {
        "total": len(test_data),
        "correct_positive": 0,
        "correct_excluded": 0,
        "correct_both": 0,
        "model_times": [],
        "llm_times": [],
    }

    for i, item in enumerate(test_data):
        query = item["query"]
        true_positive = item["positive_query"]
        true_excluded = set(item["excluded_terms"])

        print(f"[{i + 1}/{len(test_data)}] {query}")

        # Parse with fine-tuned model
        start = time.time()
        model_result = parser.parse(query)
        model_time = time.time() - start
        results["model_times"].append(model_time)

        model_positive = model_result["positive_query"]
        model_excluded = set(model_result["excluded_terms"])

        # Parse with LLM (for comparison, optional)
        start = time.time()
        llm_result = parse_with_llm(query, llm_model)
        llm_time = time.time() - start
        results["llm_times"].append(llm_time)

        # Compare
        positive_match = model_positive.lower().strip() == true_positive.lower().strip()
        excluded_match = model_excluded == true_excluded

        if positive_match:
            results["correct_positive"] += 1
        if excluded_match:
            results["correct_excluded"] += 1
        if positive_match and excluded_match:
            results["correct_both"] += 1

        print(
            f"  Model:  pos='{model_positive}' excluded={list(model_excluded)} ({model_time * 1000:.1f}ms)"
        )
        print(f"  True:   pos='{true_positive}' excluded={list(true_excluded)}")
        print(f"  Match:  positive={positive_match}, excluded={excluded_match}")
        print()

    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal samples: {results['total']}")
    print(
        f"Correct positive_query: {results['correct_positive']}/{results['total']} ({results['correct_positive'] / results['total'] * 100:.1f}%)"
    )
    print(
        f"Correct excluded_terms: {results['correct_excluded']}/{results['total']} ({results['correct_excluded'] / results['total'] * 100:.1f}%)"
    )
    print(
        f"Correct both: {results['correct_both']}/{results['total']} ({results['correct_both'] / results['total'] * 100:.1f}%)"
    )

    print(f"\nSpeed comparison:")
    print(
        f"  Model avg: {sum(results['model_times']) / len(results['model_times']) * 1000:.2f}ms"
    )
    print(
        f"  LLM avg:   {sum(results['llm_times']) / len(results['llm_times']) * 1000:.2f}ms"
    )
    print(
        f"  Speedup:   {sum(results['llm_times']) / sum(results['model_times']):.1f}x"
    )

    return results


def parse_with_llm(query, model="nemotron-3-nano:4b"):
    """Parse query using LLM for comparison."""
    prompt = f"""Parse this query and return JSON:
Query: "{query}"
Output: {{"positive_query": "...", "excluded_terms": ["..."]}}
"""
    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        return {
            "positive_query": parsed.get("positive_query", query),
            "excluded_terms": parsed.get("excluded_terms", []),
        }
    except Exception as e:
        return {"positive_query": query, "excluded_terms": []}


if __name__ == "__main__":
    import sys

    test_data_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset.json"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "./negation_model"

    evaluate_model(test_data_path, model_path)
