"""
Fine-tune FLAN-T5-small untuk negation parsing.
Approach: Seq2Seq — input query → output JSON string langsung.
Tidak ada BIO tagging, tidak ada masalah rekonstruksi subword.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
import os

MODEL_NAME = "google/flan-t5-small"  # ~300MB, cukup ringan
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128


class NegationSeq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input
        input_enc = self.tokenizer(
            item["input"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target (output JSON)
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                item["output"],
                max_length=MAX_OUTPUT_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = target_enc["input_ids"].squeeze()
        # Ganti padding token id dengan -100 supaya tidak dihitung di loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def compute_metrics(eval_preds, tokenizer):
    """
    Hitung exact match: apakah JSON output-nya identik?
    Plus JSON parse rate: apakah output valid JSON?
    """
    preds, labels = eval_preds

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels (ganti -100 dulu)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    exact_matches = 0
    json_valid = 0
    field_matches = {"positive_query": 0, "excluded_terms": 0, "iso_code": 0}

    for pred, label in zip(decoded_preds, decoded_labels):
        # Exact match
        if pred.strip() == label.strip():
            exact_matches += 1

        # JSON validity + field-level accuracy
        try:
            pred_json = json.loads(pred.strip())
            json_valid += 1

            label_json = json.loads(label.strip())

            # Cek per field
            if pred_json.get("positive_query", "").strip().lower() == \
               label_json.get("positive_query", "").strip().lower():
                field_matches["positive_query"] += 1

            if sorted(pred_json.get("excluded_terms", [])) == \
               sorted(label_json.get("excluded_terms", [])):
                field_matches["excluded_terms"] += 1

            if pred_json.get("iso_code") == label_json.get("iso_code"):
                field_matches["iso_code"] += 1

        except json.JSONDecodeError:
            pass

    n = len(decoded_preds)
    return {
        "exact_match": exact_matches / n,
        "json_valid_rate": json_valid / n,
        "positive_query_acc": field_matches["positive_query"] / n,
        "excluded_terms_acc": field_matches["excluded_terms"] / n,
        "iso_code_acc": field_matches["iso_code"] / n,
    }


def train_model(
    dataset_path: str = "synthetic_dataset.json",
    output_dir: str = "./negation_model_t5",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
):
    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    train_dataset = NegationSeq2SeqDataset(train_data, tokenizer)
    test_dataset = NegationSeq2SeqDataset(test_data, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        predict_with_generate=True,       # Penting untuk seq2seq eval
        generation_max_length=MAX_OUTPUT_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="json_valid_rate",
        logging_steps=10,
        report_to=["none"],
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating...")
    results = trainer.evaluate()
    print(f"Results: {results}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")

    return model, tokenizer


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./negation_model_t5"
    train_model(dataset_path=dataset_path, output_dir=output_dir)