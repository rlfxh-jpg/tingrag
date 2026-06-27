import argparse
import sys

sys.path.append(".")
sys.path.append("tinyrag/searcher/reranker")

from reanker_bge_m3 import RerankerBGEM3


DEFAULT_MODEL_PATH = "models/bge-reranker-base"
DEFAULT_QUERY = "机器学习是人工智能的一个分支，关注从数据中学习。"
DEFAULT_CANDIDATES = [
    "Python是一种广泛使用的高级编程语言。",
    "人工智能是计算机科学的一个分支，旨在创建智能机器。",
    "机器学习是人工智能的一个子集，主要关注从数据中学习。",
    "自然语言处理涉及计算机与人类语言的交互。",
]


def parse_candidates(raw_candidates):
    if not raw_candidates:
        return DEFAULT_CANDIDATES
    return raw_candidates


def main():
    parser = argparse.ArgumentParser(description="Run a local BGE reranker demo.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local BGE reranker model.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Query text for reranking.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        default=None,
        help="Candidate text. Repeat this flag to pass multiple candidates.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of reranked results to return.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, for example cpu or cuda.",
    )

    args = parser.parse_args()
    candidates = parse_candidates(args.candidates)

    ranker = RerankerBGEM3(model_id_key=args.model, device=args.device)
    results = ranker.rank(args.query, candidates, top_n=args.top_n)

    print("query:", args.query)
    for score, text in results:
        print(f"score={float(score):.6f} text={text}")


if __name__ == "__main__":
    main()
