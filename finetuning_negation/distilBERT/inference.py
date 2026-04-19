"""
Inference script for fine-tuned negation parsing model.
Replaces LLM-based parsing with fast model inference.
"""

import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

# Label mapping (same as training)
ID2LABEL = {
    0: "O",
    1: "B-POS",
    2: "I-POS",
    3: "B-NEG",
    4: "I-NEG",
}


class NegationParser:
    """Fast negation parser using fine-tuned model."""

    def __init__(self, model_path="./negation_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {model_path}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded. Using device: {self.device}")

    def parse(self, query: str, confidence_threshold: float = 0.7) -> dict:
        """
        Parse query to extract positive_query and excluded_terms.

        Returns:
            {
                "positive_query": str,
                "excluded_terms": list,
                "confidence": float,
            }
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        offset_mapping = inputs.pop("offset_mapping")

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values.squeeze().tolist()

        # Convert to labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
        labels = [ID2LABEL[p] for p in predictions]

        # Extract positive query and excluded terms
        positive_tokens = []
        excluded_tokens = []

        offset_mapping = offset_mapping.squeeze().tolist()

        for i, (label, token) in enumerate(zip(labels, tokens)):
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Get original character positions
            start, end = offset_mapping[i]
            if start == 0 and end == 0:
                continue

            word = query[start:end]

            if label in ["B-POS", "I-POS"]:
                positive_tokens.append((word, i))
            elif label in ["B-NEG", "I-NEG"]:
                excluded_tokens.append((word, i))

        # Reconstruct positive query
        positive_query = self._reconstruct_text(positive_tokens)

        # Reconstruct excluded terms (may be multiple)
        excluded_terms = self._extract_terms(excluded_tokens)

        # Calculate average confidence for relevant tokens
        relevant_confidences = [
            confidences[i]
            for i, label in enumerate(labels)
            if label not in ["O"] and tokens[i] not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        avg_confidence = (
            sum(relevant_confidences) / len(relevant_confidences)
            if relevant_confidences
            else 0.0
        )

        result = {
            "positive_query": positive_query if positive_query else query,
            "excluded_terms": excluded_terms,
            "confidence": avg_confidence,
        }

        return result

    def _reconstruct_text(self, tokens_with_idx):
        """Reconstruct text from tokens with indices."""
        if not tokens_with_idx:
            return ""

        # Sort by index
        tokens_with_idx.sort(key=lambda x: x[1])

        # Join tokens (handle subwords)
        words = []
        for word, _ in tokens_with_idx:
            if word.startswith("##"):
                # Subword, append to last word
                if words:
                    words[-1] += word[2:]
                else:
                    words.append(word[2:])
            else:
                words.append(word)

        return " ".join(words)

    def _extract_terms(self, tokens_with_idx):
        """Extract separate excluded terms from tokens."""
        if not tokens_with_idx:
            return []

        # Sort by index
        tokens_with_idx.sort(key=lambda x: x[1])

        # Group consecutive tokens into terms
        terms = []
        current_term = []
        prev_idx = None

        for word, idx in tokens_with_idx:
            # Check if this token is consecutive with previous
            if prev_idx is None or idx == prev_idx + 1:
                current_term.append(word)
            else:
                # New term
                if current_term:
                    terms.append(self._clean_term(" ".join(current_term)))
                current_term = [word]

            prev_idx = idx

        # Don't forget last term
        if current_term:
            terms.append(self._clean_term(" ".join(current_term)))

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())

        return unique_terms

    def _clean_term(self, term):
        """Clean reconstructed term."""
        # Handle subwords
        words = term.split()
        cleaned = []

        for word in words:
            if word.startswith("##"):
                if cleaned:
                    cleaned[-1] += word[2:]
                else:
                    cleaned.append(word[2:])
            else:
                cleaned.append(word)

        return " ".join(cleaned)


def benchmark_parser(test_queries, parser):
    """Benchmark the parser."""
    import time

    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    times = []
    results = []

    for query in test_queries:
        start = time.time()
        result = parser.parse(query)
        end = time.time()

        times.append(end - start)
        results.append(result)

        print(f"\nQuery: {query}")
        print(f"  Positive: {result['positive_query']}")
        print(f"  Excluded: {result['excluded_terms']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Time: {(end - start) * 1000:.2f}ms")

    print("\n" + "-" * 60)
    print(f"Average inference time: {sum(times) / len(times) * 1000:.2f}ms")
    print(f"Total queries: {len(test_queries)}")


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "tariffs on textiles excluding silk and cotton",
        "agricultural products excluding rice",
        "rules of origin without cumulation",
        "customs procedures not including documentation",
        "Vietnam investment provisions except financial services",
        "trade remedies",
        "dispute settlement not about arbitration",
    ]

    # Load parser
    parser = NegationParser("./negation_model")

    # Run benchmark
    benchmark_parser(test_queries, parser)
