import argparse
import sys

sys.path.append(".")
sys.path.append("tinyrag/llm")

from gguf_llm import GGUFLLM


DEFAULT_MODEL_NAME = "qwythos-local"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


def main():
    parser = argparse.ArgumentParser(description="Run a GGUF LLM demo against a local OpenAI-compatible server.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name exposed by the local inference server.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL, for example http://127.0.0.1:11434/v1.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional bearer token for the local inference server.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请介绍一下北京",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    args = parser.parse_args()

    llm = GGUFLLM(
        model_id_key=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    response = llm.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("prompt:", args.prompt)
    print("output:", response)


if __name__ == "__main__":
    main()
