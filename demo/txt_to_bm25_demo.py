import argparse
import os
import shutil
import sys

sys.path.append(".")
sys.path.append("tinyrag/parser")
sys.path.append("tinyrag/searcher/bm25_recall")
sys.path.append("tinyrag/embedding")

from txt_parser import TXTParser
from bm25_retriever import BM25Retriever


DEFAULT_FILE_PATH = "data/醉枕江山.txt"
DEFAULT_BASE_DIR = "demo_bm25_db"


def main():
    parser = argparse.ArgumentParser(
        description="Parse a txt file, chunk it, and build/save a BM25 index."
    )
    parser.add_argument("--file", type=str, default=DEFAULT_FILE_PATH, help="Input txt file path.")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="Directory used to save BM25 processed data.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Recursive chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Recursive chunk overlap.")
    parser.add_argument("--max-chunks", type=int, default=3000, help="Maximum number of chunks to keep.")
    parser.add_argument("--query", type=str, default="太平公主", help="Query used to verify BM25 retrieval.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N retrieval results to print.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing BM25 output directory before building.")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Input file does not exist: {args.file}")

    if args.reset and os.path.exists(args.base_dir):
        shutil.rmtree(args.base_dir)

    txt_parser = TXTParser(
        file_path=args.file,
        model=None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    parsed_items = txt_parser.parse()
    if not parsed_items:
        raise RuntimeError("No chunks were parsed from the input file.")

    text_chunks = [item["content"] for item in parsed_items]

    retriever = BM25Retriever(base_dir=args.base_dir)
    retriever.build(text_chunks)
    retriever.save_bm25_data()

    loaded = BM25Retriever(base_dir=args.base_dir)
    loaded.load_bm25_data()
    result = loaded.search(args.query, top_n=args.top_n)

    print(f"saved_base_dir: {args.base_dir}")
    print(f"loaded_chunks: {len(text_chunks)} / total_chunks: {len(parsed_items)}")
    print(f"query: {args.query}")
    for idx, text, score in result:
        print(f"id={idx} score={score}")
        print(text)
        print()


if __name__ == "__main__":
    main()
