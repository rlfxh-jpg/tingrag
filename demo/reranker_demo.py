import argparse
import os
import sys

sys.path.append(".")
sys.path.append("tinyrag/searcher/bm25_recall")
sys.path.append("tinyrag/searcher/emb_recall")
sys.path.append("tinyrag/searcher/reranker")
sys.path.append("tinyrag/embedding")
sys.path.append("tinyrag/llm")

from bm25_retriever import BM25Retriever
from emb_retriever import EmbRetriever
from reanker_bge_m3 import RerankerBGEM3
from hf_emb import HFSTEmbedding
from gguf_llm import GGUFLLM


DEFAULT_BM25_BASE_DIR = "demo_bm25_db"
DEFAULT_EMB_BASE_DIR = "test_milvus"
DEFAULT_EMB_MODEL_PATH = "models/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL_PATH = "models/bge-reranker-base"
DEFAULT_LLM_MODEL_NAME = "qwythos-local"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MILVUS_COLLECTION = "index_512"
DEFAULT_QUERY = "杨帆第一次破处是在哪里"
DEFAULT_TOP_N = 5
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_TOKENS = 65536
DEFAULT_TEMPERATURE = 0.3
DEFAULT_FUSION_K = 60
DEFAULT_BM25_WEIGHT = 1.2
DEFAULT_EMB_WEIGHT = 1.0
DEFAULT_RERANKER_WEIGHT = 1.5
DEFAULT_SYSTEM_PROMPT = "你是一个严谨的中文 RAG 问答助手。"
DEFAULT_RAG_PROMPT_TEMPLATE = """请根据下面的参考资料回答用户问题。

要求：
1. 只使用参考资料中能支持的信息回答，不要编造。
2. 如果参考资料不足以回答，请直接说明“根据现有参考资料无法确定”。
3. 回答要简洁、准确，优先给出结论，再补充必要依据。

用户问题：
{question}

参考资料：
{context}

请给出最终答案："""


def build_rag_prompt(query, rerank_result):
    context_lines = []
    for index, item in enumerate(rerank_result, start=1):
        score, text = item
        context_lines.append(f"[{index}] score={float(score):.6f}\n{text}")
    context = "\n\n".join(context_lines)
    return DEFAULT_RAG_PROMPT_TEMPLATE.format(question=query, context=context)


def combine_recall_results(
    bm25_recall_list,
    emb_recall_list,
    rerank_result,
    top_n,
    fusion_k=DEFAULT_FUSION_K,
    bm25_weight=DEFAULT_BM25_WEIGHT,
    emb_weight=DEFAULT_EMB_WEIGHT,
    reranker_weight=DEFAULT_RERANKER_WEIGHT,
):
    text_to_item = {}

    def ensure_item(text):
        if text not in text_to_item:
            text_to_item[text] = {
                "text": text,
                "fusion_score": 0.0,
                "bm25_score": None,
                "emb_score": None,
                "reranker_score": None,
                "bm25_rank": None,
                "emb_rank": None,
                "reranker_rank": None,
            }
        return text_to_item[text]

    for rank, (_idx, text, score) in enumerate(bm25_recall_list, start=1):
        item = ensure_item(text)
        item["bm25_score"] = score
        item["bm25_rank"] = rank
        item["fusion_score"] += bm25_weight / (fusion_k + rank)

    for rank, (_idx, text, score) in enumerate(emb_recall_list, start=1):
        item = ensure_item(text)
        item["emb_score"] = score
        item["emb_rank"] = rank
        item["fusion_score"] += emb_weight / (fusion_k + rank)

    for rank, (score, text) in enumerate(rerank_result, start=1):
        item = ensure_item(text)
        item["reranker_score"] = float(score)
        item["reranker_rank"] = rank
        item["fusion_score"] += reranker_weight / (fusion_k + rank)

    fused_items = sorted(
        text_to_item.values(),
        key=lambda item: item["fusion_score"],
        reverse=True,
    )
    return fused_items[:top_n]


def format_fused_result_for_prompt(fused_result):
    return [(item["fusion_score"], item["text"]) for item in fused_result]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a BM25 + embedding + reranker demo.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Legacy base directory containing bm_corpus and faiss_idx subdirectories.",
    )
    parser.add_argument(
        "--bm25-base-dir",
        type=str,
        default=None,
        help="Directory containing BM25 data built by txt_to_bm25_demo.py.",
    )
    parser.add_argument(
        "--emb-base-dir",
        type=str,
        default=None,
        help="Directory containing embedding recall data built by test_emb_recall_milvus.py.",
    )
    parser.add_argument(
        "--emb-model",
        type=str,
        default=DEFAULT_EMB_MODEL_PATH,
        help="Path to the local embedding model.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=DEFAULT_RERANKER_MODEL_PATH,
        help="Path to the local BGE reranker model.",
    )
    parser.add_argument(
        "--milvus-collection",
        type=str,
        default=DEFAULT_MILVUS_COLLECTION,
        help="Milvus collection name for embedding recall.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL_NAME,
        help="Model name exposed by the local OpenAI-compatible inference server.",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help="OpenAI-compatible base URL, for example http://127.0.0.1:11434/v1.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="Optional bearer token for the local inference server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query text for recall and reranking.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of reranked results to return.",
    )
    parser.add_argument(
        "--fusion-k",
        type=int,
        default=DEFAULT_FUSION_K,
        help="RRF fusion constant. Larger values make rank differences smoother.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        help="Weight for BM25 recall rank in fusion scoring.",
    )
    parser.add_argument(
        "--emb-weight",
        type=float,
        default=DEFAULT_EMB_WEIGHT,
        help="Weight for embedding recall rank in fusion scoring.",
    )
    parser.add_argument(
        "--reranker-weight",
        type=float,
        default=DEFAULT_RERANKER_WEIGHT,
        help="Weight for reranker rank in fusion scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Inference device, for example cpu or cuda.",
    )

    args = parser.parse_args(argv)
    if args.base_dir:
        args.bm25_base_dir = args.bm25_base_dir or os.path.join(args.base_dir, "bm_corpus")
        args.emb_base_dir = args.emb_base_dir or os.path.join(args.base_dir, "faiss_idx")
    else:
        args.bm25_base_dir = args.bm25_base_dir or DEFAULT_BM25_BASE_DIR
        args.emb_base_dir = args.emb_base_dir or DEFAULT_EMB_BASE_DIR
    return args


def main():
    args = parse_args()

    bm25_retriever = BM25Retriever(base_dir=args.bm25_base_dir)
    bm25_retriever.load_bm25_data()

    emb_model = HFSTEmbedding(path=args.emb_model)
    emb_retriever = EmbRetriever(
        index_dim=512,
        base_dir=args.emb_base_dir,
        collection_name=args.milvus_collection,
    )
    emb_retriever.load()

    ranker = RerankerBGEM3(model_id_key=args.reranker_model, device=args.device)
    llm = GGUFLLM(
        model_id_key=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
    )

    bm25_recall_list = bm25_retriever.search(args.query, 2 * args.top_n)
    query_emb = emb_model.get_embedding(args.query)
    emb_recall_list = emb_retriever.search(query_emb, 2 * args.top_n)

    recall_unique_text = set()
    for idx, text, score in bm25_recall_list:
        recall_unique_text.add(text)
    for idx, text, score in emb_recall_list:
        recall_unique_text.add(text)

    rerank_result = ranker.rank(args.query, list(recall_unique_text), 2 * args.top_n)
    fused_result = combine_recall_results(
        bm25_recall_list,
        emb_recall_list,
        rerank_result,
        args.top_n,
        fusion_k=args.fusion_k,
        bm25_weight=args.bm25_weight,
        emb_weight=args.emb_weight,
        reranker_weight=args.reranker_weight,
    )
    rag_prompt = build_rag_prompt(args.query, format_fused_result_for_prompt(fused_result))
    llm_answer = llm.generate(
        rag_prompt,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("query:", args.query)
    print("\n=== BM25 Recall ===")
    for idx, text, score in bm25_recall_list:
        print(f"id={idx} score={score}")
        print(text)
        print()

    print("\n=== Embedding Recall ===")
    for idx, text, score in emb_recall_list:
        print(f"id={idx} score={score}")
        print(text)
        print()

    print("\n=== Reranker Result ===")
    for score, text in rerank_result:
        print(f"score={float(score):.6f}")
        print(text)
        print()

    print("\n=== Fused Result ===")
    for item in fused_result:
        print(
            "fusion_score={:.6f} bm25_rank={} emb_rank={} reranker_rank={} "
            "bm25_score={} emb_score={} reranker_score={}".format(
                item["fusion_score"],
                item["bm25_rank"],
                item["emb_rank"],
                item["reranker_rank"],
                item["bm25_score"],
                item["emb_score"],
                item["reranker_score"],
            )
        )
        print(item["text"])
        print()

    print("\n=== LLM Prompt ===")
    print(rag_prompt)

    print("\n=== LLM Answer ===")
    print(llm_answer)


if __name__ == "__main__":
    main()
