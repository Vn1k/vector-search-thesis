"""
evaluate_model.py
=================
Evaluasi lengkap model negasi trade query menggunakan dataset original.
Jalankan di Google Colab setelah training selesai.

Usage:
    python evaluate_model.py
    python evaluate_model.py --model_dir ./negation_model_t5 --dataset synthetic_dataset.json
"""

import json
import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────── CONFIG ────────────────────────────────────────
DEFAULT_MODEL_DIR  = './negation_model_t5_v2'  # ganti sesuai path kamu
DEFAULT_DATASET    = './synthetic_dataset.json'
MAX_INPUT_LENGTH   = 128
MAX_OUTPUT_LENGTH  = 128
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
# ───────────────────────────────────────────────────────────────────────────


# ─────────────────────────── PARSER ────────────────────────────────────────
def parse_output(text: str) -> dict:
    """Parse raw output model menjadi dict terstruktur."""
    result = {'positive_query': '', 'excluded_terms': [], 'iso_code': None}
    parts  = [p.strip() for p in text.split('|')]
    for part in parts:
        if part.startswith('positive:'):
            result['positive_query'] = part[len('positive:'):].strip()
        elif part.startswith('excluded:'):
            val = part[len('excluded:'):].strip()
            if val.lower() != 'none' and val:
                result['excluded_terms'] = [t.strip() for t in val.split(',') if t.strip()]
        elif part.startswith('country:'):
            val = part[len('country:'):].strip()
            result['iso_code'] = None if val.lower() == 'none' else val
    return result
# ───────────────────────────────────────────────────────────────────────────


def load_model(model_dir: str):
    print(f'📦 Memuat model dari: {model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    print(f'   Device  : {DEVICE}')
    print(f'   Params  : {sum(p.numel() for p in model.parameters()):,}')
    return tokenizer, model


def predict(text: str, tokenizer, model) -> str:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─────────────────────────── METRICS ───────────────────────────────────────
def evaluate(dataset: list, tokenizer, model) -> dict:
    records   = []
    per_group = defaultdict(lambda: {'total': 0, 'exact': 0, 'pq': 0, 'ex': 0, 'iso': 0})

    start = time.time()
    for i, item in enumerate(dataset):
        inp    = item['input']
        meta   = item['meta']
        label  = item['output']

        raw_pred = predict(inp, tokenizer, model)
        pred     = parse_output(raw_pred)
        gold     = parse_output(label)

        exact_match  = raw_pred.strip() == label.strip()
        pq_match     = pred['positive_query'].lower().strip() == gold['positive_query'].lower().strip()
        ex_match     = sorted([t.lower() for t in pred['excluded_terms']]) == \
                       sorted([t.lower() for t in gold['excluded_terms']])
        iso_match    = pred['iso_code'] == gold['iso_code']

        records.append({
            'idx'         : i,
            'query'       : meta['query'],
            'input'       : inp,
            'label_raw'   : label,
            'pred_raw'    : raw_pred,
            'label_parsed': gold,
            'pred_parsed' : pred,
            'exact_match' : exact_match,
            'pq_match'    : pq_match,
            'ex_match'    : ex_match,
            'iso_match'   : iso_match,
        })

        # Grup berdasarkan tipe sample
        grp = 'with_country'   if meta['iso_code'] else \
              'multi_excluded'  if len(meta['excluded_terms']) > 1 else \
              'no_excluded'     if len(meta['excluded_terms']) == 0 else \
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


# ─────────────────────────── FAILURE ANALYSIS ──────────────────────────────
def analyze_failures(records: list) -> dict:
    failures = {
        'exact_only'   : [],   # salah exact tapi field-nya benar semua
        'pq_wrong'     : [],   # positive query salah
        'excluded_wrong': [],  # excluded terms salah
        'iso_wrong'    : [],   # iso code salah
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


# ─────────────────────────── PRINT REPORT ──────────────────────────────────
def print_report(metrics: dict, failures: dict, records: list):
    n = metrics['total']

    print('\n' + '═' * 60)
    print('          HASIL EVALUASI LENGKAP')
    print('═' * 60)
    print(f"  Total sampel   : {n}")
    print(f"  Waktu inferensi: {metrics['elapsed_sec']:.1f}s "
          f"({metrics['elapsed_sec']/n:.2f}s/sample)")
    print()
    print('  ── Metrik Keseluruhan ──────────────────────')
    print(f"  Exact Match          : {metrics['exact_match']:.1%}")
    print(f"  Positive Query Acc   : {metrics['positive_query_acc']:.1%}")
    print(f"  Excluded Terms Acc   : {metrics['excluded_terms_acc']:.1%}")
    print(f"  ISO Code Acc         : {metrics['iso_code_acc']:.1%}")

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
    total_wrong_exact = n - int(metrics['exact_match'] * n)
    print(f"  Total tidak exact match  : {total_wrong_exact}")
    print(f"  └─ Format beda, field OK : {len(failures['exact_only'])}  "
          f"(hanya spasi/kapital berbeda)")
    print(f"  Positive query salah     : {len(failures['pq_wrong'])}")
    print(f"  Excluded terms salah     : {len(failures['excluded_wrong'])}")
    print(f"  ISO code salah           : {len(failures['iso_wrong'])}")

    # Contoh kegagalan excluded terms
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


# ─────────────────────────── SAVE RESULTS ──────────────────────────────────
def save_results(metrics: dict, records: list, output_dir: str = '.'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Simpan summary
    summary_path = out / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'💾 Summary disimpan: {summary_path}')

    # Simpan detail per sampel
    detail_path = out / 'eval_detail.json'
    with open(detail_path, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    print(f'💾 Detail disimpan : {detail_path}')

    # Simpan hanya kegagalan
    failures_path = out / 'eval_failures.json'
    failed = [r for r in records if not r['exact_match']]
    with open(failures_path, 'w') as f:
        json.dump(failed, f, indent=2, ensure_ascii=False, default=str)
    print(f'💾 Failures disimpan: {failures_path} ({len(failed)} sampel)')


# ─────────────────────────── MAIN ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--dataset',   default=DEFAULT_DATASET)
    parser.add_argument('--output_dir', default='./eval_results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Batasi jumlah sampel (untuk debug cepat)')
    args = parser.parse_args()

    # Load dataset
    print(f'📂 Memuat dataset: {args.dataset}')
    with open(args.dataset) as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[:args.limit]
    print(f'   {len(dataset)} sampel dimuat')

    # Load model
    tokenizer, model = load_model(args.model_dir)

    # Evaluasi
    print(f'\n🔍 Mulai evaluasi ({len(dataset)} sampel)...')
    metrics, records = evaluate(dataset, tokenizer, model)

    # Analisis kegagalan
    failures = analyze_failures(records)

    # Cetak laporan
    print_report(metrics, failures, records)

    # Simpan hasil
    save_results(metrics, records, args.output_dir)


if __name__ == '__main__':
    main()