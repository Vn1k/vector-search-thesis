#!/bin/bash

# Cek apakah user memasukkan argumen query
if [ -z "$1" ]; then
  echo "❌ Error: Query tidak boleh kosong."
  echo "💡 Cara pakai: ./test_search.sh \"masukkan query Anda di sini\""
  exit 1
fi

QUERY=$1

echo "🚀 Mengirim request ke Backend..."
echo "🔗 Endpoint : POST http://localhost:8000/api/v1/search"
echo "🔍 Query    : '$QUERY'"
echo "--------------------------------------------------"

curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\"}" | jq .

echo ""
echo "--------------------------------------------------"
echo "✅ Request selesai. Silakan cek log di terminal Backend Anda."