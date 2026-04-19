import argparse
from dataclasses import dataclass

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import settings

try:
    import localspelling
except ImportError:  # pragma: no cover - fallback untuk environment tanpa localspelling
    localspelling = None


@dataclass
class TruncationResult:
    full_tokens: int
    used_tokens: int
    is_truncated: bool
    cut_char_idx: int
    kept_text: str
    dropped_text: str


def clean_preview(text: str, limit: int = 220) -> str:
    one_line = " ".join(str(text).split())
    if len(one_line) <= limit:
        return one_line
    return f"{one_line[:limit]}..."


def build_context_text(row: pd.Series) -> str:
    agreement_name = row["Agreement"]
    chapter_name = row["chapter_name"] if pd.notna(row["chapter_name"]) else "General Provisions"
    article_name = row["article_name"] if pd.notna(row["article_name"]) else ""
    article_text = row["article_text"] if pd.notna(row["article_text"]) else ""
    return f"{agreement_name} - {chapter_name} - {article_name}: {article_text}"


def load_dataset() -> pd.DataFrame:
    dt_cover = pd.read_csv(settings.FILE_COVER)
    dt_detail = pd.read_csv(settings.FILE_DETAIL)

    df = pd.merge(dt_detail, dt_cover, left_on="rta_id", right_on="id", how="inner")
    df = df.dropna(subset=["article_text"])
    df = df.drop_duplicates(subset=["article_text", "Agreement"])

    df["context_length"] = (
        df["Agreement"].astype(str)
        + " - "
        + df["chapter_name"].astype(str)
        + " - "
        + df["article_name"].astype(str)
        + ": "
        + df["article_text"].astype(str)
    ).str.len()
    return df


def get_tokenizer(model: SentenceTransformer):
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        return model.tokenizer
    first_module = model._first_module()
    if hasattr(first_module, "tokenizer"):
        return first_module.tokenizer
    raise RuntimeError("Tokenizer tidak ditemukan di SentenceTransformer model.")


def analyze_truncation(text: str, tokenizer, max_seq_length: int) -> TruncationResult:
    full = tokenizer(text, add_special_tokens=True, truncation=False)
    full_input_ids = full["input_ids"]
    full_tokens = len(full_input_ids)

    truncated = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=getattr(tokenizer, "is_fast", False),
    )
    used_tokens = len(truncated["input_ids"])
    is_truncated = full_tokens > max_seq_length

    cut_char_idx = len(text)
    if is_truncated:
        if "offset_mapping" in truncated:
            offsets = truncated["offset_mapping"]
            spans = [pair for pair in offsets if pair[1] > pair[0]]
            if spans:
                cut_char_idx = spans[-1][1]
        else:
            decoded = tokenizer.decode(
                truncated["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            cut_char_idx = text.find(decoded)
            if cut_char_idx >= 0:
                cut_char_idx += len(decoded)
            else:
                cut_char_idx = len(decoded)

    kept_text = text[:cut_char_idx]
    dropped_text = text[cut_char_idx:] if cut_char_idx < len(text) else ""

    return TruncationResult(
        full_tokens=full_tokens,
        used_tokens=used_tokens,
        is_truncated=is_truncated,
        cut_char_idx=cut_char_idx,
        kept_text=kept_text,
        dropped_text=dropped_text,
    )


def boundary_preview(text: str, cut_idx: int, radius: int = 120) -> str:
    left = max(0, cut_idx - radius)
    right = min(len(text), cut_idx + radius)
    window = text[left:right]
    rel = cut_idx - left
    marked = f"{window[:rel]}[CUT]{window[rel:]}"
    return clean_preview(marked, limit=2 * radius + 80)


def report_for_row(
    row_index: int,
    row: pd.Series,
    tokenizer,
    max_seq_length: int,
    chunk_size: int,
    chunk_overlap: int,
    char_limit: int | None,
    preview_limit: int,
) -> None:
    raw_context = build_context_text(row)
    if localspelling is not None:
        normalized_context = localspelling.convert_spelling(raw_context, "us")
    else:
        normalized_context = raw_context
    source_text = normalized_context[:char_limit] if char_limit else normalized_context

    one_shot = analyze_truncation(source_text, tokenizer, max_seq_length)

    print("\n" + "=" * 90)
    print(f"ROW INDEX: {row_index}")
    print(f"AGREEMENT: {row['Agreement']}")
    print(f"CHAPTER  : {row['chapter_name']}")
    print(f"ARTICLE  : {row['article_name']}")
    print("-" * 90)
    print(f"Panjang teks sumber          : {len(source_text)} karakter")
    print(f"Total token (tanpa truncate) : {one_shot.full_tokens}")
    print(f"Token dipakai model          : {one_shot.used_tokens}/{max_seq_length}")
    print(f"Terpotong oleh MPNet         : {'YA' if one_shot.is_truncated else 'TIDAK'}")

    if one_shot.is_truncated:
        print(f"Karakter yang terbuang       : {len(one_shot.dropped_text)}")
        print("Boundary cut (sekitar batas):")
        print(clean_preview(boundary_preview(source_text, one_shot.cut_char_idx), preview_limit))
        print("Awal teks yang terpotong:")
        print(clean_preview(one_shot.dropped_text, preview_limit))
    else:
        print("Tidak ada teks yang terbuang di mode tanpa LangChain.")

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(source_text)
    chunk_results = [analyze_truncation(chunk, tokenizer, max_seq_length) for chunk in chunks]
    truncated_chunk_idx = [i + 1 for i, result in enumerate(chunk_results) if result.is_truncated]

    print("\n[Simulasi Dengan LangChain]")
    print(f"Jumlah chunk                 : {len(chunks)}")
    print(f"Chunk size (token)           : {chunk_size}")
    print(f"Chunk overlap (token)        : {chunk_overlap}")
    print(f"Chunk yang masih terpotong   : {len(truncated_chunk_idx)}")
    if truncated_chunk_idx:
        print(f"Daftar chunk terpotong       : {truncated_chunk_idx}")
    else:
        print("Semua chunk aman (tidak kena truncation MPNet).")

    print("Contoh preview chunk-1:")
    if chunks:
        print(clean_preview(chunks[0], preview_limit))
    else:
        print("(kosong)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tampilkan teks yang terpotong oleh batas token MPNet dan bandingkan dengan LangChain splitter."
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Index dataframe hasil merge (seperti ingest.py). Jika kosong, pakai baris terpanjang.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Jumlah baris terpanjang yang diuji jika --row-index tidak diisi.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.VECTOR_MODEL,
        help="Nama sentence-transformer model.",
    )
    parser.add_argument(
        "--char-limit",
        type=int,
        default=None,
        help="Opsional simulasi potong karakter sebelum encode (contoh: 3000 seperti ingest.py).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Ukuran chunk token untuk LangChain. Default: max_seq_length - 2.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Overlap chunk dalam token untuk LangChain splitter.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=240,
        help="Batas karakter untuk preview output di terminal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    tokenizer = get_tokenizer(model)
    max_seq_length = model.get_max_seq_length() or model.max_seq_length
    if max_seq_length is None:
        raise RuntimeError("Gagal mendapatkan max_seq_length model.")

    chunk_size = args.chunk_size if args.chunk_size is not None else max_seq_length - 2
    if chunk_size <= 0:
        raise ValueError("chunk_size harus > 0.")
    if args.chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap harus lebih kecil dari chunk_size.")

    print(f"MPNet max_seq_length: {max_seq_length} token")
    print("Loading dataset...")
    df = load_dataset()
    print(f"Total data siap uji: {len(df)}")

    if args.row_index is not None:
        if args.row_index not in df.index:
            raise IndexError(f"row-index {args.row_index} tidak ditemukan. Contoh index valid: {df.index.min()}..{df.index.max()}")
        target_indices = [args.row_index]
    else:
        target_indices = df.nlargest(args.top_n, "context_length").index.tolist()

    print(f"Target row index: {target_indices}")
    if args.char_limit:
        print(f"Simulasi char limit aktif: {args.char_limit}")

    for idx in target_indices:
        row = df.loc[idx]
        report_for_row(
            row_index=idx,
            row=row,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            chunk_size=chunk_size,
            chunk_overlap=args.chunk_overlap,
            char_limit=args.char_limit,
            preview_limit=args.preview_limit,
        )

    print("\nSelesai.")


if __name__ == "__main__":
    main()
