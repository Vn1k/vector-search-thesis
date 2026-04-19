# FLAN-T5-small for Negation Parsing

Fine-tune FLAN-T5-small untuk negation parsing dengan approach Seq2Seq.

## Keunggulan FLAN-T5 vs DistilBERT

| Aspek | DistilBERT | FLAN-T5 |
|-------|-----------|---------|
| **Task** | Token Classification | Seq2Seq Generation |
| **Output** | BIO labels per token | JSON string langsung |
| **Tokenization issues** | Ada (subword splits) | Tidak ada |
| **Post-processing** | Perlu (reconstruct text) | Tidak perlu |
| **Model size** | 66MB | ~300MB |
| **Training** | Klasifikasi token | Generation |

## Cara Running

### Step 1: Generate Dataset

```bash
cd /home/vinik/Github/vector-search-thesis/finetuning_negation/FLAN

# Generate 400 samples (seq2seq format)
python generate_dataset_FLAN.py 400 synthetic_dataset.json
```

**Output:** `synthetic_dataset.json` dengan format:
```json
[
  {
    "input": "Parse the trade agreement search query... Query: tariffs excluding rice",
    "output": "{\"positive_query\": \"tariffs\", \"excluded_terms\": [\"rice\"], \"iso_code\": null}",
    "meta": { ... }
  }
]
```

### Step 2: Training

**Option A: Training di Local (GTX 1660)**

```bash
# Training dengan batch size kecil
python train_model.py synthetic_dataset.json ./negation_model_t5
```

**Config default:**
- Batch size: 16
- Epochs: 10
- Learning rate: 3e-4
- Estimated time: ~40-60 menit

**Option B: Training di Google Colab (Recommended)**

Gunakan notebook `../colab/train_flan_t5.ipynb` (bisa buat sendiri mirip dengan yang DistilBERT).

### Step 3: Inference

Setelah training selesai, model bisa dipakai untuk inference:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Load model
model = T5ForConditionalGeneration.from_pretrained("./negation_model_t5")
tokenizer = T5TokenizerFast.from_pretrained("./negation_model_t5")

# Parse query
query = "tariffs on textiles excluding silk and cotton"
input_text = f"Parse the trade agreement search query. Extract the positive topic and any excluded terms. Query: {query}"

inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
outputs = model.generate(**inputs, max_length=128)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Result: {"positive_query": "tariffs on textiles", "excluded_terms": ["silk", "cotton"], "iso_code": null}
parsed = json.loads(result)
```

## Integrasi ke Search Engine

Tambahkan di `search_engine.py`:

```python
# Load FLAN-T5 model
from transformers import T5ForConditionalGeneration, T5TokenizerFast

class SearchService:
    def __init__(self, ...):
        # ... existing init ...
        
        # FLAN-T5 parser
        self.use_flan = True
        if self.use_flan:
            self.flan_tokenizer = T5TokenizerFast.from_pretrained("./FLAN/negation_model_t5")
            self.flan_model = T5ForConditionalGeneration.from_pretrained("./FLAN/negation_model_t5")
            
    def parse_with_flan(self, query: str) -> dict:
        input_text = f"Parse the trade agreement search query... Query: {query}"
        inputs = self.flan_tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
        outputs = self.flan_model.generate(**inputs, max_length=128)
        result = self.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return json.loads(result)
```

## Expected Results

FLAN-T5 diharapkan lebih baik karena:
- **No tokenization reconstruction** - output langsung string
- **No BIO tagging** - tidak perlu decode labels
- **Seq2Seq nature** - lebih natural untuk text-to-text task

Target metrics:
- Exact match: >75%
- JSON valid rate: >90%
- Inference time: ~20-50ms

## Troubleshooting

### CUDA Out of Memory
Kurangi batch size di `train_model.py`:
```python
batch_size=8  # dari 16
```

### Model tidak converge
Coba:
- Naikkan epochs: `epochs=15`
- Turunkan learning rate: `learning_rate=1e-4`

### Output tidak valid JSON
Ini normal di awal training. Model akan belajar generate valid JSON setelah beberapa epoch.
