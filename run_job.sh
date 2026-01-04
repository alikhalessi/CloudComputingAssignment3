#!/usr/bin/env bash
set -euo pipefail

echo "=== System Info ==="
uname -a || true
echo "=== GPU Check (should show NVIDIA T4 on ACA GPU) ==="
nvidia-smi || true
echo "==================="

: "${FRAMES_ZIP_URL:?Set FRAMES_ZIP_URL to a SAS URL for frames.zip (Blob)}"
: "${OUTPUT_BLOB_URL:?Set OUTPUT_BLOB_URL to a SAS URL for uploading model.csv (Blob)}"

VISION_MODEL="${VISION_MODEL:-llava}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

cd /app
mkdir -p data/images out

echo "Downloading frames.zip from Blob..."
curl -L --fail --retry 5 --retry-delay 2 "$FRAMES_ZIP_URL" -o /app/frames.zip

echo "Extracting frames.zip..."
rm -rf /tmp/frames && mkdir -p /tmp/frames
unzip -o /app/frames.zip -d /tmp/frames >/dev/null

echo "Flattening images into /app/data/images ..."
find /tmp/frames -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) \
  -exec cp -f {} /app/data/images/ \;

rm -rf /tmp/frames

echo "Building images.txt ..."
find /app/data/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort > /app/images.txt
echo "Images count: $(wc -l < /app/images.txt)"

echo "Starting Ollama..."
ollama serve > /tmp/ollama_runtime.log 2>&1 &
OLLAMA_PID=$!
sleep 2

echo "Ensuring models exist..."
ollama list | grep -q "$VISION_MODEL" || ollama pull "$VISION_MODEL"
ollama list | grep -q "$EMBED_MODEL" || ollama pull "$EMBED_MODEL"

echo "Running pipeline..."
python3 pipeline.py \
  --images /app/images.txt \
  --model /app/out/model.csv \
  --vision_model "$VISION_MODEL" \
  --embed_model "$EMBED_MODEL"

echo "Uploading model.csv to Blob..."
curl -X PUT --fail --retry 5 --retry-delay 2 \
  -H "x-ms-blob-type: BlockBlob" \
  --upload-file /app/out/model.csv \
  "$OUTPUT_BLOB_URL"

echo "Sample queries (logs only):"
python3 query.py --model /app/out/model.csv --question "Find images with a cat" --top_k 5 --embed_model "$EMBED_MODEL" || true
python3 query.py --model /app/out/model.csv --question "Find images with a dog" --top_k 5 --embed_model "$EMBED_MODEL" || true

BYTES="$(wc -c < /app/out/model.csv || true)"
echo "DONE. model.csv bytes: ${BYTES}"

kill "$OLLAMA_PID" || true
