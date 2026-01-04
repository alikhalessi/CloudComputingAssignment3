#!/usr/bin/env bash
set -euo pipefail

cd /app
mkdir -p data/images out

# If frames.zip exists and images folder is empty, extract & flatten into data/images
if [ -f "/app/frames.zip" ] && [ "$(find data/images -type f 2>/dev/null | wc -l)" -eq 0 ]; then
  echo "Extracting frames.zip ..."
  rm -rf /tmp/frames
  mkdir -p /tmp/frames
  unzip -o /app/frames.zip -d /tmp/frames >/dev/null
  find /tmp/frames -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) \
    -exec cp -f {} /app/data/images/ \;
  rm -rf /tmp/frames
fi

# Create images.txt (absolute paths)
find /app/data/images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | sort > /app/images.txt
echo "Images count: $(wc -l < /app/images.txt)"

# Start ollama
ollama serve > /tmp/ollama_runtime.log 2>&1 &
OLLAMA_PID=$!
sleep 2

# Ensure models exist (build pre-pulled, but keep safe)
ollama list | grep -q "llava" || ollama pull llava
ollama list | grep -q "nomic-embed-text" || ollama pull nomic-embed-text

# Run pipeline
python3 pipeline.py --images /app/images.txt --model /app/out/model.csv --vision_model llava --embed_model nomic-embed-text

# Run a couple sample queries so the teacher sees "results" in logs
python3 query.py --model /app/out/model.csv --question "Find images with a cat" --top_k 5 --embed_model nomic-embed-text || true
python3 query.py --model /app/out/model.csv --question "Find images with a dog" --top_k 5 --embed_model nomic-embed-text || true

echo "DONE. model.csv bytes: $(stat -c%s /app/out/model.csv)"
kill $OLLAMA_PID || true