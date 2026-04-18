"""
evaluate_distilbert.py
======================
Evaluasi lengkap DistilBERT negation parsing menggunakan dataset BIO.
Struktur laporan identik dengan evaluate_flan.py untuk perbandingan yang fair.

Usage:
    python evaluate_distilbert.py
    python evaluate_distilbert.py --model_dir ./negation_model_distilbert --dataset synthetic_dataset_distilbert.json
    python evaluate_distilbert.py --model_dir ./negation_model_distilbert_raw --raw_subword
"""

import json
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification  # ← BEDA: bukan Seq2SeqLM

# ─────────────────────────── CONFIG ────────────────────────────────────────
DEFAULT_MODEL_DIR = './negation_model_distilbert'
DEFAULT_DATASET   = './synthetic_dataset_distilbert.json'
MAX_LENGTH        = 128
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'

LABEL2ID = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'B-NEG': 3, 'I-NEG': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── REKONSTRUKSI ──────────────────────────────────
# BEDA dari FLAN: tidak ada parse_output() string — rekonstruksi dari BIO tags

def reconstruct_word_ids(word_tokens: list, pred_label_ids: list,
                         word_ids: list) -> dict:
    """
    Rekonstruksi menggunakan word_ids() — versi FIX.
    Ambil prediksi dari subword PERTAMA tiap kata saja.
    Hasil: kata tetap utuh, tidak ada 're med ies'.
    """
    word_preds = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_preds:  # hanya subword pertama
            word_preds[word_id] = ID2LABEL.get(pred_label_ids[token_idx], 'O')

    pred_labels = [word_preds.get(i, 'O') for i in range(len(word_tokens))]
    return _bio_to_dict(word_tokens, pred_labels)


def reconstruct_raw_subword(tokens: list, pred_label_ids: list) -> dict:
    """
    Rekonstruksi via prefix ## — versi RAW (tanpa word_ids fix).
    Subword ke-2, ke-3 digabung berdasarkan prefix ##.
    Dipakai saat --raw_subword flag aktif.
    """
    pos_parts   = []
    neg_chunks  = []
    current_neg = []

    for token, label_id in zip(tokens, pred_label_ids):
        if token in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        label       = ID2LABEL.get(label_id, 'O')
        is_subword  = token.startswith('##')
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
        'excluded_terms': neg_chunks,
        'iso_code':       None,  # DistilBERT tidak predict iso_code
    }


def _bio_to_dict(word_tokens: list, pred_labels: list) -> dict:
    """Helper: konversi list (word, label) ke dict hasil."""
    pos_words   = []
    neg_chunks  = []
    current_neg = []

    for word, label in zip(word_tokens, pred_labels):
        if label in ('B-POS', 'I-POS'):
            pos_words.append(word)
        elif label == 'B-NEG':
            if current_neg:
                neg_chunks.append(' '.join(current_neg))
            current_neg = [word]
        elif label == 'I-NEG':
            current_neg.append(word)
        else:
            if current_neg:
                neg_chunks.append(' '.join(current_neg))
                current_neg = []

    if current_neg:
        neg_chunks.append(' '.join(current_neg))

    return {
        'positive_query': ' '.join(pos_words),
        'excluded_terms': neg_chunks,
        'iso_code':       None,  # DistilBERT tidak predict iso_code
    }
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── LOAD MODEL ────────────────────────────────────
def load_model(model_dir: str):
    print(f'📦 Memuat model dari: {model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)  # ← BEDA
    model.eval()
    print(f'   Device  : {DEVICE}')
    print(f'   Params  : {sum(p.numel() for p in model.parameters()):,}')
    print(f'   Labels  : {model.config.id2label}')
    return tokenizer, model
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── PREDICT ───────────────────────────────────────
# BEDA dari FLAN: bukan model.generate() — melainkan logits.argmax()

def predict(word_tokens: list, tokenizer, model, use_raw_subword: bool = False) -> dict:
    """
    Inference DistilBERT.
    Returns dict: {positive_query, excluded_terms, iso_code, pred_labels, token_acc_data}
    """
    if use_raw_subword:
        # Versi RAW: tokenize sebagai string biasa
        encoding = tokenizer(
            ' '.join(word_tokens),
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)
    else:
        # Versi FIX: tokenize per kata dengan word_ids
        encoding = tokenizer(
            word_tokens,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)

    with torch.no_grad():
        logits = model(**encoding).logits  # [1, seq_len, num_labels]

    pred_ids = logits.argmax(dim=-1).squeeze().tolist()

    if use_raw_subword:
        tokens = tokenizer.convert_ids_to_tokens(
            encoding['input_ids'].squeeze().tolist()
        )
        result = reconstruct_raw_subword(tokens, pred_ids)
        result['_pred_ids'] = pred_ids
        result['_tokens']   = tokens
    else:
        word_ids = encoding.word_ids(batch_index=0)
        result   = reconstruct_word_ids(word_tokens, pred_ids, word_ids)
        result['_pred_ids'] = pred_ids
        result['_word_ids'] = word_ids

    return result
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── METRICS ───────────────────────────────────────
def evaluate(dataset: list, tokenizer, model, use_raw_subword: bool = False) -> tuple:
    records   = []
    per_group = defaultdict(lambda: {'total': 0, 'exact': 0, 'pq': 0, 'ex': 0, 'iso': 0})

    start = time.time()
    for i, item in enumerate(dataset):
        word_tokens = item['word_tokens']
        bio_labels  = item['bio_labels']
        meta        = item.get('meta', item)  # fallback ke item sendiri

        # Gold langsung dari field dataset — tidak perlu parse string seperti FLAN
        gold = {
            'positive_query': item.get('positive_query', ''),
            'excluded_terms': item.get('excluded_terms', []),
            'iso_code':       item.get('iso_code', None),
        }

        pred = predict(word_tokens, tokenizer, model, use_raw_subword)

        # ── Field-level match ──────────────────────────────────────────
        pq_match  = pred['positive_query'].lower().strip() == \
                    gold['positive_query'].lower().strip()
        ex_match  = sorted([t.lower() for t in pred['excluded_terms']]) == \
                    sorted([t.lower() for t in gold['excluded_terms']])
        iso_match = True  # DistilBERT tidak predict iso_code, anggap selalu benar

        # exact_match untuk DistilBERT = semua field benar
        # (tidak ada "raw string" seperti FLAN untuk dibandingkan)
        exact_match = pq_match and ex_match

        # ── Token-level accuracy (label vs gold BIO) ───────────────────
        # Hanya untuk info tambahan, tidak ada di FLAN
        token_acc = _compute_token_acc(
            word_tokens, bio_labels, pred['_pred_ids'],
            pred.get('_word_ids'), pred.get('_tokens'),
            use_raw_subword, tokenizer
        )

        records.append({
            'idx'          : i,
            'query'        : item['query'],
            'word_tokens'  : word_tokens,
            'bio_labels'   : bio_labels,
            'label_parsed' : gold,
            'pred_parsed'  : pred,
            'pred_raw'     : f"positive: {pred['positive_query']} | "
                             f"excluded: {', '.join(pred['excluded_terms']) or 'none'} | "
                             f"country: none",
            'label_raw'    : f"positive: {gold['positive_query']} | "
                             f"excluded: {', '.join(gold['excluded_terms']) or 'none'} | "
                             f"country: {gold['iso_code'] or 'none'}",
            'exact_match'  : exact_match,
            'pq_match'     : pq_match,
            'ex_match'     : ex_match,
            'iso_match'    : iso_match,
            'token_acc'    : token_acc,
        })

        # Grup berdasarkan tipe sample — identik dengan FLAN
        grp = 'with_country'  if meta.get('iso_code') else \
              'multi_excluded' if len(gold['excluded_terms']) > 1 else \
              'no_excluded'    if len(gold['excluded_terms']) == 0 else \
              'single_excluded'

        per_group[grp]['total'] += 1
        per_group[grp]['exact'] += int(exact_match)
        per_group[grp]['pq']    += int(pq_match)
        per_group[grp]['ex']    += int(ex_match)
        per_group[grp]['iso']   += int(iso_match)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f'   [{i+1}/{len(dataset)}] {elapsed:.1f}s elapsed')

    elapsed = time.time() - start
    n       = len(records)

    metrics = {
        'total'             : n,
        'elapsed_sec'       : round(elapsed, 2),
        'exact_match'       : sum(r['exact_match'] for r in records) / n,
        'positive_query_acc': sum(r['pq_match']    for r in records) / n,
        'excluded_terms_acc': sum(r['ex_match']    for r in records) / n,
        'iso_code_acc'      : sum(r['iso_match']   for r in records) / n,
        'token_accuracy'    : sum(r['token_acc']   for r in records) / n,  # extra metric DistilBERT
        'per_group'         : {},
    }

    for grp, g in per_group.items():
        t = g['total']
        metrics['per_group'][grp] = {
            'total'             : t,
            'exact_match'       : g['exact'] / t,
            'positive_query_acc': g['pq']    / t,
            'excluded_terms_acc': g['ex']    / t,
            'iso_code_acc'      : g['iso']   / t,
        }

    return metrics, records


def _compute_token_acc(word_tokens, bio_labels, pred_ids,
                       word_ids, tokens, use_raw_subword, tokenizer) -> float:
    """Hitung token-level accuracy vs gold BIO label."""
    try:
        if use_raw_subword:
            # Align gold label ke subword via char offset
            query       = ' '.join(word_tokens)
            encoding    = tokenizer(query, truncation=True, max_length=MAX_LENGTH,
                                    return_offsets_mapping=True)
            offsets     = encoding['offset_mapping']
            query_lower = query.lower()
            char_label  = {}
            cursor      = 0
            for word, bio in zip(word_tokens, bio_labels):
                idx = query_lower.find(word.lower(), cursor)
                if idx == -1:
                    continue
                for c in range(idx, idx + len(word)):
                    char_label[c] = LABEL2ID.get(bio, 0)
                cursor = idx + len(word)
            gold_ids = [char_label.get(s, 0) if not (s == 0 and e == 0) else -100
                        for s, e in offsets]
        else:
            # Align gold label ke subword via word_ids
            gold_ids = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    gold_ids.append(-100)
                elif wid != prev_wid:
                    gold_ids.append(LABEL2ID.get(bio_labels[wid], 0))
                else:
                    gold_ids.append(-100)
                prev_wid = wid

        correct = total = 0
        for p, g in zip(pred_ids[:len(gold_ids)], gold_ids):
            if g == -100:
                continue
            correct += int(p == g)
            total   += 1
        return correct / total if total > 0 else 0.0
    except Exception:
        return 0.0
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── FAILURE ANALYSIS ──────────────────────────────
# Identik dengan FLAN — tidak ada perubahan
def analyze_failures(records: list) -> dict:
    failures = {
        'exact_only'    : [],
        'pq_wrong'      : [],
        'excluded_wrong': [],
        'iso_wrong'     : [],
    }
    for r in records:
        if not r['exact_match']:
            all_fields_ok = r['pq_match'] and r['ex_match'] and r['iso_match']
            if all_fields_ok:
                failures['exact_only'].append(r)
            if not r['pq_match']:
                failures['pq_wrong'].append(r)
            if not r['ex_match']:
                failures['excluded_wrong'].append(r)
            if not r['iso_match']:
                failures['iso_wrong'].append(r)
    return failures
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── PRINT REPORT ──────────────────────────────────
# Identik dengan FLAN + tambahan token_accuracy
def print_report(metrics: dict, failures: dict, records: list):
    n = metrics['total']

    print('\n' + '═' * 60)
    print('     HASIL EVALUASI LENGKAP — DistilBERT BIO')
    print('═' * 60)
    print(f"  Total sampel   : {n}")
    print(f"  Waktu inferensi: {metrics['elapsed_sec']:.1f}s "
          f"({metrics['elapsed_sec']/n:.2f}s/sample)")
    print()
    print('  ── Metrik Keseluruhan ──────────────────────')
    print(f"  Exact Match          : {metrics['exact_match']:.1%}  (pq + excl benar)")
    print(f"  Positive Query Acc   : {metrics['positive_query_acc']:.1%}")
    print(f"  Excluded Terms Acc   : {metrics['excluded_terms_acc']:.1%}")
    print(f"  ISO Code Acc         : {metrics['iso_code_acc']:.1%}  (N/A — tidak diprediksi)")
    print(f"  Token Accuracy       : {metrics['token_accuracy']:.1%}  (extra metric BIO)")

    print()
    print('  ── Per-Grup ────────────────────────────────')
    group_order = ['single_excluded', 'multi_excluded', 'no_excluded', 'with_country']
    group_label = {
        'single_excluded': 'Single excluded  ',
        'multi_excluded' : 'Multi excluded   ',
        'no_excluded'    : 'No excluded      ',
        'with_country'   : 'With country     ',
    }
    header = f"  {'Grup':<22} {'N':>4}  {'Exact':>6}  {'PosQ':>6}  {'Excl':>6}  {'ISO':>6}"
    print(header)
    print('  ' + '-' * 56)
    for grp in group_order:
        if grp not in metrics['per_group']:
            continue
        g = metrics['per_group'][grp]
        print(f"  {group_label[grp]:<22} {g['total']:>4}  "
              f"{g['exact_match']:>5.1%}  {g['positive_query_acc']:>5.1%}  "
              f"{g['excluded_terms_acc']:>5.1%}  {g['iso_code_acc']:>5.1%}")

    print()
    print('  ── Analisis Kegagalan ──────────────────────')
    total_wrong = n - int(metrics['exact_match'] * n)
    print(f"  Total tidak exact match  : {total_wrong}")
    print(f"  └─ Format beda, field OK : {len(failures['exact_only'])}")
    print(f"  Positive query salah     : {len(failures['pq_wrong'])}")
    print(f"  Excluded terms salah     : {len(failures['excluded_wrong'])}")
    print(f"  ISO code salah           : {len(failures['iso_wrong'])}")

    if failures['excluded_wrong']:
        print()
        print('  ── Contoh Kegagalan Excluded Terms (maks 3) ──')
        for r in failures['excluded_wrong'][:3]:
            print(f"  Query  : {r['query']}")
            print(f"  Gold   : {r['label_raw']}")
            print(f"  Pred   : {r['pred_raw']}")
            print()

    if failures['pq_wrong']:
        print()
        print('  ── Contoh Kegagalan Positive Query (maks 3) ──')
        for r in failures['pq_wrong'][:3]:
            print(f"  Query  : {r['query']}")
            print(f"  Gold   : {r['label_raw']}")
            print(f"  Pred   : {r['pred_raw']}")
            print()

    print('═' * 60)
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── SAVE RESULTS ──────────────────────────────────
# Identik dengan FLAN
def save_results(metrics: dict, records: list, output_dir: str = '.'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'💾 Summary disimpan : {summary_path}')

    detail_path = out / 'eval_detail.json'
    with open(detail_path, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    print(f'💾 Detail disimpan  : {detail_path}')

    failures_path = out / 'eval_failures.json'
    failed = [r for r in records if not r['exact_match']]
    with open(failures_path, 'w') as f:
        json.dump(failed, f, indent=2, ensure_ascii=False, default=str)
    print(f'💾 Failures disimpan: {failures_path} ({len(failed)} sampel)')
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── MAIN ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',  default=DEFAULT_MODEL_DIR)
    parser.add_argument('--dataset',    default=DEFAULT_DATASET)
    parser.add_argument('--output_dir', default='./eval_results_distilbert')
    parser.add_argument('--limit',      type=int, default=None,
                        help='Batasi jumlah sampel (untuk debug cepat)')
    parser.add_argument('--raw_subword', action='store_true',
                        help='Gunakan rekonstruksi raw subword (tanpa word_ids fix)')
    args = parser.parse_args()

    print(f'📂 Memuat dataset: {args.dataset}')
    with open(args.dataset) as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]
    print(f'   {len(dataset)} sampel dimuat')

    mode = 'RAW subword (tanpa fix)' if args.raw_subword else 'word_ids fix'
    print(f'   Mode rekonstruksi: {mode}')

    tokenizer, model = load_model(args.model_dir)

    print(f'\n🔍 Mulai evaluasi ({len(dataset)} sampel)...')
    metrics, records = evaluate(dataset, tokenizer, model, args.raw_subword)

    failures = analyze_failures(records)
    print_report(metrics, failures, records)
    save_results(metrics, records, args.output_dir)


if __name__ == '__main__':
    main()