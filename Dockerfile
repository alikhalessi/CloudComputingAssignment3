FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git unzip curl ca-certificates dos2unix \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN dos2unix /app/run_job.sh || true && chmod +x /app/run_job.sh

ENV OLLAMA_HOST=0.0.0.0:11434

ENTRYPOINT ["/app/run_job.sh"]