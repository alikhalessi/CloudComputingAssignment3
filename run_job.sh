#!/usr/bin/env bash
set -euo pipefail

cd /app
mkdir -p data/images out

echo "=== A1 Job starting ==="
echo "Working dir: $(pwd)"
echo "Python: $(python3 --version || true)"
echo "Pip: $(python3 -m pip --version || true)"

# ---- A1: Download frames.zip from Azure Blob via SAS URL ----
# You must pass FRAMES_URL as an environment variable to the Container Apps Job
if [ ! -f "/app/frames.zip" ] && [ -n "${FRAMES_URL:-}" ]; then
  echo "Downloading frames.zip from Blob..."
  curl -L "$FRAMES_URL" -o /app/frames.zip
fi

# ---- Proof of GPU (if present) ----
echo "GPU devices check:"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* seen"
nvidia-smi 2>/dev/null || true

# ---- Extract frames.zip into data/images (flatten) if images folder empty ----
if [ -f "/app/frames.zip" ] && [ "$(find /app/data/images -type f 2>/dev/null | wc -l)" -eq 0 ]; then
  echo "Extracting frames.zip ..."
  rm -rf /tmp/frames
  mkdir -p /tmp/frames
  unzip -o /app/frames.zip -d /tmp/frames >/dev/null

  # Copy/flatten all images into /app/data/images
  find /tmp/frames -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) \
    -exec cp -f {} /app/data/images/ \;

  rm -rf /tmp/frames
fi

# ---- Create images.txt (absolute paths) ----
find /app/data/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort > /app/images.txt
echo "Images count: $(wc -l < /app/images.txt)"

# ---- Start Ollama server ----
echo "Starting ollama serve..."
ollama serve > /tmp/ollama_runtime.log 2>&1 &
OLLAMA_PID=$!
sleep 2

# ---- Ensure models exist (pull at runtime) ----
echo "Ensuring models are present..."
ollama list | grep -qi "llava" || ollama pull llava
ollama list | grep -qi "nomic-embed-text" || ollama pull nomic-embed-text

# ---- Run pipeline ----
echo "Running pipeline..."
python3 pipeline.py \
  --images /app/images.txt \
  --model /app/out/model.csv \
  --vision_model llava \
  --embed_model nomic-embed-text

# ---- Sample queries (shows evidence in logs) ----
echo "Running sample queries..."
python3 query.py --model /app/out/model.csv --question "Find images with a cat" --top_k 5 --embed_model nomic-embed-text || true
python3 query.py --model /app/out/model.csv --question "Find images with a dog" --top_k 5 --embed_model nomic-embed-text || true

# ---- Upload model.csv back to Blob (so job output persists) ----
# You must pass RESULTS_URL as an environment variable to the Container Apps Job
if [ -n "${RESULTS_URL:-}" ] && [ -f "/app/out/model.csv" ]; then
  echo "Uploading out/model.csv to Blob..."
  curl -X PUT -T /app/out/model.csv -H "x-ms-blob-type: BlockBlob" "$RESULTS_URL"
fi

echo "DONE. model.csv bytes: $(stat -c%s /app/out/model.csv 2>/dev/null || wc -c < /app/out/model.csv)"

# ---- Stop ollama ----
kill $OLLAMA_PID 2>/dev/null || true
echo "=== A1 Job finished ==="
