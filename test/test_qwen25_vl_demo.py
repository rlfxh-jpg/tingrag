import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from demo.qwen25_vl_image_caption_demo import build_image_rag_text, parse_caption_json


def test_parse_caption_json_strips_markdown_fence():
    raw = """```json
{
  "image_type": "产品原型图",
  "summary": "登录页截图",
  "visible_text": ["登录", "手机号"]
}
```"""

    parsed = parse_caption_json(raw)

    assert parsed["image_type"] == "产品原型图"
    assert parsed["summary"] == "登录页截图"
    assert parsed["visible_text"] == ["登录", "手机号"]


def test_parse_caption_json_falls_back_to_summary():
    parsed = parse_caption_json("这是一张登录页截图")

    assert parsed["image_type"] == "其他"
    assert parsed["summary"] == "这是一张登录页截图"
    assert "模型输出不是严格 JSON" in parsed["uncertain"][0]


def test_build_image_rag_text_includes_source_context_and_caption():
    caption = {
        "image_type": "产品原型图",
        "summary": "订单详情页待付款状态",
        "visible_text": ["订单详情", "立即支付"],
        "ui_elements": ["底部操作栏"],
        "business_rules": ["待付款状态展示立即支付按钮"],
        "user_actions": ["立即支付"],
        "states": ["待付款"],
        "table_data": "",
        "uncertain": ["商品图看不清"],
    }

    rag_text = build_image_rag_text(
        caption=caption,
        image_path="data/doc_images/order/page_8_img_1.png",
        source_file="订单需求说明.docx",
        page=8,
        context="本节说明订单详情页状态。",
    )

    assert "资料类型：需求文档图片" in rag_text
    assert "订单需求说明.docx" in rag_text
    assert "第 8 页" in rag_text
    assert "订单详情页待付款状态" in rag_text
    assert "待付款状态展示立即支付按钮" in rag_text
    assert "data/doc_images/order/page_8_img_1.png" in rag_text
