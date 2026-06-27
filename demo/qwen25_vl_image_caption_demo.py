import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


CAPTION_KEYS = {
    "image_type": "其他",
    "summary": "",
    "visible_text": [],
    "ui_elements": [],
    "business_rules": [],
    "user_actions": [],
    "states": [],
    "table_data": "",
    "uncertain": [],
}


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned


def parse_caption_json(text: str) -> Dict[str, Any]:
    cleaned = strip_markdown_fence(text)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("caption JSON must be an object")
    except Exception:
        data = {
            "image_type": "其他",
            "summary": cleaned,
            "visible_text": [],
            "ui_elements": [],
            "business_rules": [],
            "user_actions": [],
            "states": [],
            "table_data": "",
            "uncertain": ["模型输出不是严格 JSON，需要人工检查或二次清洗"],
        }

    normalized = dict(CAPTION_KEYS)
    normalized.update(data)

    for key in ["visible_text", "ui_elements", "business_rules", "user_actions", "states", "uncertain"]:
        normalized[key] = ensure_list(normalized.get(key))

    normalized["image_type"] = str(normalized.get("image_type") or "其他")
    normalized["summary"] = str(normalized.get("summary") or "")
    normalized["table_data"] = str(normalized.get("table_data") or "")
    return normalized


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(value)]


def join_items(items: Any) -> str:
    values = ensure_list(items)
    return "、".join(values)


def build_image_rag_text(
    caption: Dict[str, Any],
    image_path: str,
    source_file: str = "",
    page: Optional[int] = None,
    context: str = "",
    ocr_text: str = "",
) -> str:
    page_text = f"第 {page} 页" if page is not None else "未知"
    lines = [
        "资料类型：需求文档图片",
        f"来源文件：{source_file or '未知'}",
        f"页码：{page_text}",
        f"图片路径：{image_path}",
        "",
        f"文档上下文：{context or '无'}",
        f"图片类型：{caption.get('image_type', '其他')}",
        f"图片摘要：{caption.get('summary', '')}",
        f"图片可见文字：{join_items(caption.get('visible_text'))}",
        f"OCR文字：{ocr_text or '无'}",
        f"页面元素：{join_items(caption.get('ui_elements'))}",
        f"业务规则：{join_items(caption.get('business_rules'))}",
        f"用户操作：{join_items(caption.get('user_actions'))}",
        f"状态信息：{join_items(caption.get('states'))}",
        f"表格内容：{caption.get('table_data', '') or '无'}",
        f"不确定信息：{join_items(caption.get('uncertain'))}",
    ]
    return "\n".join(lines).strip()


def build_prompt(context: str = "") -> str:
    return f"""
你是一个需求文档图片解析助手。请分析这张需求文档中的图片。

要求：
1. 只根据图片中可见内容和给定上下文回答。
2. 不要编造看不见的信息。
3. 如果不确定，写入 uncertain 字段。
4. 输出严格 JSON，不要输出 Markdown，不要额外解释。

JSON 字段：
{{
  "image_type": "产品截图 / 原型图 / 流程图 / 表格截图 / 架构图 / 其他",
  "summary": "一句话概括图片内容",
  "visible_text": ["图片中可见的主要文字"],
  "ui_elements": ["页面中的主要控件、区域、按钮、字段"],
  "business_rules": ["能从图片和上下文中明确看出的业务规则"],
  "user_actions": ["用户可以执行的操作"],
  "states": ["页面状态、按钮状态、订单状态、流程状态等"],
  "table_data": "如果是表格，尽量转成 Markdown 表格；否则为空字符串",
  "uncertain": ["不确定或看不清的信息"]
}}

文档上下文：
{context or "无"}
""".strip()


def load_chat_template(model_id: str, tokenizer: Any) -> str:
    template_path = Path(model_id) / "chat_template.json"
    if template_path.exists():
        data = json.loads(template_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("chat_template"), str):
            return data["chat_template"]
    return tokenizer.chat_template


def load_qwen25_vl_processor(model_id: str, min_pixels: int, max_pixels: int):
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels * 28 * 28,
            max_pixels=max_pixels * 28 * 28,
            use_fast=False,
        )
    except Exception as exc:
        try:
            from transformers import (
                AutoTokenizer,
                Qwen2VLImageProcessor,
                Qwen2VLVideoProcessor,
                Qwen2_5_VLProcessor,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            image_processor = Qwen2VLImageProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels * 28 * 28,
                max_pixels=max_pixels * 28 * 28,
            )
            video_processor = Qwen2VLVideoProcessor.from_pretrained(model_id)
            return Qwen2_5_VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                chat_template=load_chat_template(model_id, tokenizer),
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "加载 Qwen2.5-VL processor 失败。AutoProcessor 和手动组装 processor 都未成功。"
            ) from fallback_exc


def describe_image_with_qwen25_vl(
    image_path: str,
    context: str = "",
    model_id: str = DEFAULT_MODEL,
    min_pixels: int = 256,
    max_pixels: int = 1024,
    max_new_tokens: int = 768,
) -> Dict[str, Any]:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "缺少 Qwen2.5-VL 推理依赖。请先安装 README 里的 transformers、torch、qwen-vl-utils、pillow。"
        ) from exc

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = load_qwen25_vl_processor(
        model_id=model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_file)},
                {"type": "text", "text": build_prompt(context)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parse_caption_json(output_text)


def write_text(path: Optional[str], text: str) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use Qwen2.5-VL-3B to describe a requirement-document image.")
    parser.add_argument("--image", required=True, help="需求文档图片路径，例如 data/doc_images/demo.png")
    parser.add_argument("--context", default="", help="图片前后文或章节说明")
    parser.add_argument("--source-file", default="", help="原始需求文档文件名")
    parser.add_argument("--page", type=int, default=None, help="图片所在页码")
    parser.add_argument("--ocr-text", default="", help="可选 OCR 文字，拼入 RAG 文本")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 ID，默认 {DEFAULT_MODEL}")
    parser.add_argument("--min-pixels", type=int, default=256, help="视觉 token 下限系数，实际传入值为该值 * 28 * 28")
    parser.add_argument("--max-pixels", type=int, default=1024, help="视觉 token 上限系数，实际传入值为该值 * 28 * 28")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="最大输出 token")
    parser.add_argument("--output", default="", help="可选，保存结构化 JSON 的路径")
    parser.add_argument("--rag-output", default="", help="可选，保存 RAG 文本块的路径")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    caption = describe_image_with_qwen25_vl(
        image_path=args.image,
        context=args.context,
        model_id=args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    rag_text = build_image_rag_text(
        caption=caption,
        image_path=args.image,
        source_file=args.source_file,
        page=args.page,
        context=args.context,
        ocr_text=args.ocr_text,
    )

    json_text = json.dumps(caption, ensure_ascii=False, indent=2)
    print("=== Caption JSON ===")
    print(json_text)
    print("\n=== RAG Text ===")
    print(rag_text)

    write_text(args.output, json_text)
    write_text(args.rag_output, rag_text)


if __name__ == "__main__":
    main()
