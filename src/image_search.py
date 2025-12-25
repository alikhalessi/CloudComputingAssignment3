import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import ollama


def pick_vision_model_default() -> str:
    # Slides mention ministral-3:3b; allow override via env var
    return os.getenv("VISION_MODEL", "ministral-3:3b")


# Step 1: prompt-on-image -> text
def describe_image(image_path: str, prompt: str, model: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_bytes],
        }],
    )
    return resp["message"]["content"].strip()


# Step 2: embedding of string -> numpy array (normalized)
def embed_text(text: str, model: str = "embeddinggemma") -> np.ndarray:
    resp = ollama.embeddings(model=model, prompt=text)
    vec = np.array(resp["embedding"], dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return vec if n == 0.0 else (vec / n)


# Step 3: “model” dataframe
def new_model_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["filename", "description", "embedding"])


# Step 4: similarity search
@dataclass
class SearchResult:
    filename: str
    description: str
    score: float


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def search_most_similar(model_df: pd.DataFrame, query_embedding: np.ndarray, top_k: int = 1) -> List[SearchResult]:
    if len(model_df) == 0:
        return []

    scores = model_df["embedding"].apply(lambda v: cosine_sim(query_embedding, v))
    best_idx = np.argsort(-scores.values)[:top_k]

    results: List[SearchResult] = []
    for i in best_idx:
        row = model_df.iloc[i]
        results.append(SearchResult(
            filename=str(row["filename"]),
            description=str(row["description"]),
            score=float(scores.iloc[i]),
        ))
    return results


# Step 5: save/load CSV
def save_model_csv(model_df: pd.DataFrame, csv_path: str) -> None:
    out = model_df.copy()
    out["embedding"] = out["embedding"].apply(lambda v: json.dumps([float(x) for x in v]))
    out.to_csv(csv_path, index=False)


def load_model_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
    df["embedding"] = df["embedding"].apply(lambda v: v / (np.linalg.norm(v) + 1e-12))
    return df


# Step 9 helper: add images to model
def add_images(
    model_df: pd.DataFrame,
    image_paths: List[str],
    prompt: str,
    vision_model: str,
    embed_model: str = "embeddinggemma",
    skip_existing: bool = True,
) -> pd.DataFrame:
    existing = set(model_df["filename"].tolist()) if (skip_existing and len(model_df) > 0) else set()

    rows = []
    for p in image_paths:
        if skip_existing and p in existing:
            continue
        desc = describe_image(p, prompt=prompt, model=vision_model)
        emb = embed_text(desc, model=embed_model)
        rows.append({"filename": p, "description": desc, "embedding": emb})

    if not rows:
        return model_df

    return pd.concat([model_df, pd.DataFrame(rows)], ignore_index=True)
