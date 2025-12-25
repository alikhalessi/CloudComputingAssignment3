from src.image_search import load_model_csv, embed_text, search_most_similar


def main():
    model_path = "model_full.csv"
    df = load_model_csv(model_path)

    print("Model loaded. Type a question. Type 'exit' to quit.")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        qe = embed_text(q, model="embeddinggemma")
        r = search_most_similar(df, qe, top_k=1)[0]
        print(f"\nBEST MATCH: {r.filename}")
        print(f"DESC: {r.description}")
        print(f"SCORE: {r.score:.4f}")


if __name__ == "__main__":
    main()
