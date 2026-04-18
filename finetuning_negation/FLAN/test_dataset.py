import json

with open("synthetic_dataset.json") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Cek format sudah pipe-delimited (bukan JSON)
sample = data[0]
print(f"\nSample input  : {sample['input']}")
print(f"Sample output : {sample['output']}")

# Pastikan output formatnya benar
assert "positive:" in sample["output"], "❌ Format salah — masih JSON lama?"
assert "excluded:" in sample["output"], "❌ Format salah — masih JSON lama?"
print("\n✅ Format dataset OK, siap training")