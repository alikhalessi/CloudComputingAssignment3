import argparse
import os
import sys

from src.image_search import add_images, load_model_csv, new_model_df, save_model_csv, pick_vision_model_default


def read_image_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description="Batch pipeline: process images and update model CSV.")
    ap.add_argument("--images", required=True, help="Text file with one image filename per line")
    ap.add_argument("--model", required=True, help="CSV model file (created if missing)")
    ap.add_argument("--prompt", default="Describe this image in 1-2 sentences.")
    ap.add_argument("--vision_model", default=pick_vision_model_default(),
                    help="Ollama model for image description (must support images)")
    ap.add_argument("--embed_model", default="embeddinggemma")
    ap.add_argument("--no_skip_existing", action="store_true")
    args = ap.parse_args()

    image_paths = [p for p in read_image_list(args.images) if os.path.exists(p)]
    if not image_paths:
        print("No valid image paths found.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(args.model):
        df = load_model_csv(args.model)
    else:
        df = new_model_df()

    df = add_images(
        df,
        image_paths=image_paths,
        prompt=args.prompt,
        vision_model=args.vision_model,
        embed_model=args.embed_model,
        skip_existing=not args.no_skip_existing,
    )

    save_model_csv(df, args.model)
    print(f"Saved model: {args.model}")
    print(f"Rows in model: {len(df)}")


if __name__ == "__main__":
    main()
