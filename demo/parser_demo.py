import argparse
import os
import sys
from pathlib import Path

sys.path.append(".")
sys.path.append("tinyrag/parser")
sys.path.append("tinyrag/embedding")

from hf_emb import HFSTEmbedding
from img_emb import ImgEmbedding
from txt_parser import TXTParser
from md_parser import MDParser
from img_parser import ImgParser

try:
    from pdf_parser import PDFParser
except Exception:
    PDFParser = None

try:
    from doc_parser import WordParser
except Exception:
    WordParser = None

try:
    from ppt_parser import PPTXParser
except Exception:
    PPTXParser = None


IMAGE_TYPES = {"png", "jpg", "jpeg"}
PARSER_MAP = {
    "txt": TXTParser,
    "md": MDParser,
    "image": ImgParser,
}
if PDFParser is not None:
    PARSER_MAP["pdf"] = PDFParser
if WordParser is not None:
    PARSER_MAP["docx"] = WordParser
if PPTXParser is not None:
    PARSER_MAP["pptx"] = PPTXParser


def resolve_suffix(file_path: str, suffix_override: str | None):
    suffix = suffix_override.strip(".").lower() if suffix_override else Path(file_path).suffix.strip(".").lower()
    if suffix in IMAGE_TYPES:
        return "image"
    return suffix


def parser_file(file_path: str, model, suffix: str | None):
    resolved_suffix = resolve_suffix(file_path, suffix)
    parser_cls = PARSER_MAP.get(resolved_suffix)
    if parser_cls is None:
        supported = ", ".join(sorted(PARSER_MAP.keys()))
        raise NotImplementedError(f"Suffix '{resolved_suffix}' is not supported in this environment. Supported: {supported}")
    return parser_cls(file_path, model).parse()


def build_model(file_path: str, image_types, use_embedding: bool, text_model: str, image_model: str):
    if not use_embedding:
        return None

    suffix = Path(file_path).suffix.strip(".").lower()
    if suffix in image_types:
        return ImgEmbedding(path=image_model)
    return HFSTEmbedding(path=text_model)


def preview_item(item):
    preview = {
        "file_path": item.get("file_path"),
        "page": item.get("page"),
        "title": item.get("title"),
        "author": item.get("author"),
        "subject": item.get("subject"),
        "content": item.get("content"),
    }
    emb = item.get("embedding")
    if emb is None:
        preview["embedding"] = None
    else:
        preview["embedding_dim"] = len(emb)
        preview["embedding_head"] = emb[:5]
    return preview


def main():
    parser = argparse.ArgumentParser(description="Parse a document with the project's parser module.")
    parser.add_argument("--file", type=str, required=True, help="Input file path.")
    parser.add_argument("--suffix", type=str, default=None, help="Optional parser suffix override, for example txt or pdf.")
    parser.add_argument("--text-model", type=str, default="models/bge-small-zh-v1.5", help="Text embedding model path.")
    parser.add_argument("--image-model", type=str, default="models/clip-ViT-B-32", help="Image embedding model path.")
    parser.add_argument("--limit", type=int, default=3, help="Number of parsed items to print.")
    parser.add_argument("--no-embedding", action="store_true", help="Parse only, do not compute embeddings.")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Input file does not exist: {args.file}")

    model = build_model(
        file_path=args.file,
        image_types=IMAGE_TYPES,
        use_embedding=not args.no_embedding,
        text_model=args.text_model,
        image_model=args.image_model,
    )

    items = parser_file(args.file, model=model, suffix=args.suffix)
    if not items:
        print("No parsed items returned.")
        return

    print(f"parsed_items: {len(items)}")
    for idx, item in enumerate(items[: args.limit], start=1):
        print(f"--- item {idx} ---")
        print(preview_item(item))


if __name__ == "__main__":
    main()
