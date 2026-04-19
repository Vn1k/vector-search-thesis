import json
import gc
import time
import os
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# ─────────────────────────── KONFIGURASI ──────────────────────────────────
# Pastikan path ini BENAR dan folder/file-nya ada!
FLAN_MODEL_DIR    = './FLAN/negation_model_t5_v2'
FLAN_DATASET      = './FLAN/synthetic_dataset.json'

DISTIL_MODEL_DIR  = './distilBERT/negation_model_distilBERT_v3'
DISTIL_DATASET    = './distilBERT/synthetic_dataset_distilbert.json'

DEVICE            = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Label DistilBERT
LABEL2ID = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'B-NEG': 3, 'I-NEG': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ──────────────────────────────────────────────────────────────────────────

def check_paths():
    """Mencegah error HuggingFace dengan mengecek keberadaan folder lokal."""
    paths = [FLAN_MODEL_DIR, FLAN_DATASET, DISTIL_MODEL_DIR, DISTIL_DATASET]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"\n❌ ERROR: Path tidak ditemukan -> '{p}'\nPastikan path sudah benar dan kamu menjalankan script dari direktori yang tepat.")

# ==========================================================================
# 1. FUNGSI MENGHITUNG F1-SCORE (WORD-LEVEL)
# ==========================================================================
def compute_f1_text(pred_text: str, gold_text: str):
    """Menghitung Precision, Recall, dan F1-Score berdasarkan irisan kata."""
    pred_tokens = pred_text.lower().split()
    gold_tokens = gold_text.lower().split()

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    
    common_counts = pred_counts & gold_counts
    overlap = sum(common_counts.values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

# ==========================================================================
# 2. LOGIKA FLAN-T5
# ==========================================================================
def parse_flan_output(text: str) -> dict:
    result = {'positive_query': '', 'excluded_terms': []}
    parts  = [p.strip() for p in text.split('|')]
    for part in parts:
        if part.startswith('positive:'):
            result['positive_query'] = part[len('positive:'):].strip()
        elif part.startswith('excluded:'):
            val = part[len('excluded:'):].strip()
            if val.lower() != 'none' and val:
                result['excluded_terms'] = [t.strip() for t in val.split(',') if t.strip()]
    return result

def eval_flan():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 MEMULAI EVALUASI FLAN-T5")
    tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_DIR).to(DEVICE)
    model.eval()

    with open(FLAN_DATASET, 'r') as f:
        dataset = json.load(f)

    metrics = {'pq_f1': [], 'ex_f1': []}
    
    print(f"Memproses {len(dataset)} sampel data...")
    for item in dataset:
        inputs = tokenizer(item['input'], return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
            
        raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = parse_flan_output(raw_pred)
        gold = parse_flan_output(item['output'])

        _, _, pq_f1 = compute_f1_text(pred['positive_query'], gold['positive_query'])
        metrics['pq_f1'].append(pq_f1)

        pred_ex_str = " ".join(pred['excluded_terms'])
        gold_ex_str = " ".join(gold['excluded_terms'])
        _, _, ex_f1 = compute_f1_text(pred_ex_str, gold_ex_str)
        metrics['ex_f1'].append(ex_f1)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        'pq_f1': sum(metrics['pq_f1']) / len(metrics['pq_f1']),
        'ex_f1': sum(metrics['ex_f1']) / len(metrics['ex_f1']),
    }

# ==========================================================================
# 3. LOGIKA DISTILBERT (TANPA RECONSTRUCTION / RAW SUBWORD)
# ==========================================================================
def reconstruct_raw_subword(tokens: list, pred_label_ids: list) -> dict:
    """Versi RAW: membaca token dan tag ## apa adanya."""
    pos_parts, neg_chunks, current_neg = [], [], []

    for token, label_id in zip(tokens, pred_label_ids):
        if token in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        label = ID2LABEL.get(label_id, 'O')
        is_subword = token.startswith('##')
        clean_token = token[2:] if is_subword else token

        if label in ('B-POS', 'I-POS'):
            if is_subword and pos_parts:
                pos_parts[-1] += clean_token
            else:
                pos_parts.append(clean_token)
        elif label == 'B-NEG':
            if current_neg:
                neg_chunks.append(' '.join(current_neg))
            current_neg = [clean_token]
        elif label == 'I-NEG':
            if is_subword and current_neg:
                current_neg[-1] += clean_token
            else:
                current_neg.append(clean_token)
        else:
            if current_neg:
                neg_chunks.append(' '.join(current_neg))
                current_neg = []

    if current_neg:
        neg_chunks.append(' '.join(current_neg))

    return {
        'positive_query': ' '.join(pos_parts),
        'excluded_terms': neg_chunks
    }

def eval_distilbert():
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 MEMULAI EVALUASI DISTILBERT (RAW)")
    tokenizer = AutoTokenizer.from_pretrained(DISTIL_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(DISTIL_MODEL_DIR).to(DEVICE)
    model.eval()

    with open(DISTIL_DATASET, 'r') as f:
        dataset = json.load(f)

    metrics = {'pq_f1': [], 'ex_f1': []}

    print(f"Memproses {len(dataset)} sampel data...")
    for item in dataset:
        word_tokens = item['word_tokens']
        gold = {
            'positive_query': item.get('positive_query', ''),
            'excluded_terms': item.get('excluded_terms', [])
        }

        # Tokenize sebagai string biasa (Raw), bukan split_into_words
        query_str = ' '.join(word_tokens)
        encoding = tokenizer(query_str, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
            logits = model(**encoding).logits

        pred_ids = logits.argmax(dim=-1).squeeze().tolist()
        
        # Ekstrak token string dari input_ids
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist())
        
        # Gunakan RAW reconstruct
        pred = reconstruct_raw_subword(tokens, pred_ids)

        _, _, pq_f1 = compute_f1_text(pred['positive_query'], gold['positive_query'])
        metrics['pq_f1'].append(pq_f1)

        pred_ex_str = " ".join(pred['excluded_terms'])
        gold_ex_str = " ".join(gold['excluded_terms'])
        _, _, ex_f1 = compute_f1_text(pred_ex_str, gold_ex_str)
        metrics['ex_f1'].append(ex_f1)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'pq_f1': sum(metrics['pq_f1']) / len(metrics['pq_f1']),
        'ex_f1': sum(metrics['ex_f1']) / len(metrics['ex_f1']),
    }

# ==========================================================================
# 4. MAIN EXECUTION
# ==========================================================================
if __name__ == '__main__':
    print("=====================================================")
    print("   MEMULAI BENCHMARK F1-SCORE: FLAN-T5 vs DistilBERT")
    print(f"   Device aktif: {DEVICE}")
    print("=====================================================")

    # Cek path dulu sebelum jalan agar tidak error di tengah-tengah!
    check_paths()

    flan_results = eval_flan()
    distil_results = eval_distilbert()

    print("\n\n" + "="*53)
    print(" 🏆 HASIL AKHIR: PERBANDINGAN F1-SCORE (WORD-LEVEL)")
    print("="*53)
    print(f"{'Metrik':<25} | {'DistilBERT':<10} | {'FLAN-T5':<10}")
    print("-" * 53)
    print(f"{'Positive Query F1':<25} | {distil_results['pq_f1']:.2%}     | {flan_results['pq_f1']:.2%}")
    print(f"{'Excluded Terms F1':<25} | {distil_results['ex_f1']:.2%}     | {flan_results['ex_f1']:.2%}")
    print("=" * 53)