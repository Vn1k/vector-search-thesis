"""
Fine-tune DistilBERT v2 for negation parsing with improvements.
- Better dataset (domain-specific topics)
- Post-processing evaluation
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Label mapping
LABELS = {
    "O": 0,
    "B-POS": 1,
    "I-POS": 2,
    "B-NEG": 3,
    "I-NEG": 4,
}
ID2LABEL = {v: k for k, v in LABELS.items()}


class NegationDataset(Dataset):
    """Dataset for negation parsing."""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        positive = item["positive_query"]
        excluded_terms = item["excluded_terms"]

        # Tokenize
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Create labels using offset mapping
        labels = self.create_labels(query, positive, excluded_terms, encoding)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def create_labels(self, query, positive, excluded_terms, encoding):
        """Create BIO labels for tokens."""
        query_lower = query.lower()
        positive_lower = positive.lower()

        # Initialize all labels as O
        labels = [LABELS["O"]] * self.max_length

        offset_mapping = encoding["offset_mapping"].squeeze().tolist()

        # Tag positive query part
        if positive_lower in query_lower:
            start_idx = query_lower.find(positive_lower)
            end_idx = start_idx + len(positive_lower)

            inside_positive = False
            for i, (start, end) in enumerate(offset_mapping[: self.max_length]):
                if start == 0 and end == 0:
                    continue

                if start >= start_idx and end <= end_idx:
                    if not inside_positive:
                        labels[i] = LABELS["B-POS"]
                        inside_positive = True
                    else:
                        labels[i] = LABELS["I-POS"]

        # Tag excluded terms
        for term in excluded_terms:
            term_lower = term.lower()
            if term_lower in query_lower:
                start_idx = query_lower.find(term_lower)
                end_idx = start_idx + len(term_lower)

                inside_term = False
                for i, (start, end) in enumerate(offset_mapping[: self.max_length]):
                    if start == 0 and end == 0:
                        continue

                    if start >= start_idx and end <= end_idx:
                        if not inside_term:
                            labels[i] = LABELS["B-NEG"]
                            inside_term = True
                        else:
                            labels[i] = LABELS["I-NEG"]

        return labels


def load_and_prepare_data(dataset_path, tokenizer, test_size=0.2):
    """Load and prepare dataset."""
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Split train/test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    train_dataset = NegationDataset(train_data, tokenizer)
    test_dataset = NegationDataset(test_data, tokenizer)

    return train_dataset, test_dataset, train_data, test_data


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Flatten
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()

    # Remove padding (-100)
    mask = labels_flat != -100
    labels_flat = labels_flat[mask]
    preds_flat = preds_flat[mask]

    accuracy = accuracy_score(labels_flat, preds_flat)

    # Classification report
    report = classification_report(
        labels_flat,
        preds_flat,
        target_names=["O", "B-POS", "I-POS", "B-NEG", "I-NEG"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


def train_model(
    dataset_path="synthetic_dataset_v2.json",
    output_dir="./negation_model_v2",
    epochs=8,
    batch_size=8,
    learning_rate=5e-5,
):
    """Train the negation parsing model."""

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABELS,
    )

    print("Loading dataset...")
    train_dataset, test_dataset, train_data, test_data = load_and_prepare_data(
        dataset_path, tokenizer
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training arguments with improvements
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Add warmup
        logging_dir="./logs_v2",
        logging_steps=10,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to=["none"],
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,  # Effective batch size 32
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset_v2.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./negation_model_v2"

    train_model(dataset_path=dataset_path, output_dir=output_dir)
