import argparse
import sys

sys.path.append(".")
sys.path.append("tinyrag/embedding")

from hf_emb import HFSTEmbedding


DEFAULT_MODEL_PATH = "models/bge-small-zh-v1.5"


def main():
    parser = argparse.ArgumentParser(description="Run a local BGE embedding demo.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local BGE sentence-transformers model.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="请介绍一下北京",
        help="Primary text to encode.",
    )
    parser.add_argument(
        "--text2",
        type=str,
        default=None,
        help="Optional second text for cosine similarity comparison.",
    )

    args = parser.parse_args()

    emb_model = HFSTEmbedding(path=args.model)
    vec1 = emb_model.get_embedding(args.text)

    print("text:", args.text)
    print("vector_type:", type(vec1).__name__)
    print("vector_dim:", len(vec1))
    print("vector_head:", vec1[:5])

    if args.text2:
        vec2 = emb_model.get_embedding(args.text2)
        similarity = emb_model.cosine_similarity(vec1, vec2)
        print("text2:", args.text2)
        print("similarity:", similarity)


if __name__ == "__main__":
    main()
