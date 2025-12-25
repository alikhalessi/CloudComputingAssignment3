import argparse
from src.image_search import load_model_csv, embed_text, search_most_similar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--embed_model", default="embeddinggemma")
    args = ap.parse_args()

    df = load_model_csv(args.model)
    qe = embed_text(args.question, model=args.embed_model)
    results = search_most_similar(df, qe, top_k=args.top_k)

    if not results:
        print("No results (empty model).")
        return

    for r in results:
        print(f"BEST MATCH: {r.filename}")
        print(f"SCORE: {r.score:.4f}")
        print(f"DESC: {r.description}")
        print("-" * 40)


if __name__ == "__main__":
    main()
