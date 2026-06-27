# Qwen2.5-VL Requirement Image Demo

这个 demo 用 `Qwen/Qwen2.5-VL-3B-Instruct` 解析需求文档里的截图、原型图、流程图或表格截图，并输出两类结果：

- 结构化 JSON：适合落库、人工检查、后处理。
- RAG 文本块：适合继续用 BGE embedding 写入现有 TinyRAG 向量库。

脚本位置：

```bash
demo/qwen25_vl_image_caption_demo.py
```

## 1. 推荐配置

最低建议：

```text
GPU：NVIDIA 8GB 显存，可以先试，但需要限制图片尺寸
内存：16GB RAM
Python：3.10 或 3.11
```

更稳配置：

```text
GPU：NVIDIA 12GB 或 16GB 显存
内存：32GB RAM
Python：3.10 或 3.11
```

CPU 也可以跑，但速度会很慢，不建议批量处理需求文档图片。

## 2. 安装依赖

建议先建一个单独环境：

```bash
conda create -n qwen25vl python=3.10 -y
conda activate qwen25vl
```

安装 PyTorch CUDA 版。下面示例使用 CUDA 12.1：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

安装 Qwen2.5-VL 相关依赖：

```bash
pip install -U transformers accelerate qwen-vl-utils pillow
```

如果你已经安装了 PyTorch，先检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出 `True` 说明 PyTorch 能看到 GPU。

## 3. 运行 Demo

准备一张需求文档图片，例如：

```text
data/doc_images/demo.png
```

运行：

```bash
python demo/qwen25_vl_image_caption_demo.py ^
  --image data/doc_images/demo.png ^
  --context "这是订单详情页需求里的截图，说明待付款状态下的页面展示。" ^
  --source-file "订单需求说明.docx" ^
  --page 8 ^
  --output demo/output/demo_caption.json ^
  --rag-output demo/output/demo_rag_text.txt
```

PowerShell 单行写法：

```powershell
python demo/qwen25_vl_image_caption_demo.py --image data/doc_images/demo.png --context "这是订单详情页需求里的截图，说明待付款状态下的页面展示。" --source-file "订单需求说明.docx" --page 8 --output demo/output/demo_caption.json --rag-output demo/output/demo_rag_text.txt
```

第一次运行会从 Hugging Face 下载模型：

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

下载会比较慢，模型缓存通常在 Hugging Face 默认缓存目录中。

## 4. 输出结果

脚本会打印结构化 JSON：

```json
{
  "image_type": "产品原型图",
  "summary": "该图片展示订单详情页的待付款状态。",
  "visible_text": ["订单详情", "待付款", "取消订单", "立即支付"],
  "ui_elements": ["订单状态区域", "商品列表", "底部操作栏"],
  "business_rules": ["待付款状态下展示取消订单和立即支付按钮"],
  "user_actions": ["取消订单", "立即支付"],
  "states": ["待付款"],
  "table_data": "",
  "uncertain": []
}
```

同时生成适合进入 RAG 的文本块：

```text
资料类型：需求文档图片
来源文件：订单需求说明.docx
页码：第 8 页
图片路径：data/doc_images/demo.png

文档上下文：这是订单详情页需求里的截图，说明待付款状态下的页面展示。
图片类型：产品原型图
图片摘要：该图片展示订单详情页的待付款状态。
图片可见文字：订单详情、待付款、取消订单、立即支付
OCR文字：无
页面元素：订单状态区域、商品列表、底部操作栏
业务规则：待付款状态下展示取消订单和立即支付按钮
用户操作：取消订单、立即支付
状态信息：待付款
表格内容：无
不确定信息：
```

后续可以把这段 RAG 文本继续交给项目里的 BGE：

```python
from tinyrag import HFSTEmbedding

bge = HFSTEmbedding(path="models/bge-small-zh-v1.5")
embedding = bge.get_embedding(rag_text)
```

## 5. 参数说明

常用参数：

```text
--image             必填，图片路径
--context           可选，图片前后文或章节说明
--source-file       可选，原始需求文档文件名
--page              可选，图片所在页码
--ocr-text          可选，OCR 文字，会拼入 RAG 文本
--model             默认 Qwen/Qwen2.5-VL-3B-Instruct
--output            可选，保存 JSON
--rag-output        可选，保存 RAG 文本
```

显存相关参数：

```text
--min-pixels        默认 256
--max-pixels        默认 1024
--max-new-tokens    默认 768
```

如果显存不够，先降低：

```bash
python demo/qwen25_vl_image_caption_demo.py --image data/doc_images/demo.png --max-pixels 768 --max-new-tokens 512
```

如果图片是复杂表格或流程图，识别不够细，可以尝试提高：

```bash
python demo/qwen25_vl_image_caption_demo.py --image data/doc_images/demo.png --max-pixels 1280 --max-new-tokens 1024
```

## 6. 常见问题

### ModuleNotFoundError: No module named 'qwen_vl_utils'

安装：

```bash
pip install -U qwen-vl-utils
```

### CUDA out of memory

优先尝试：

```bash
python demo/qwen25_vl_image_caption_demo.py --image data/doc_images/demo.png --max-pixels 768 --max-new-tokens 512
```

仍然不够时：

- 降低图片分辨率。
- 一次只处理一张图片。
- 使用量化模型，例如 AWQ 版本。
- 换 12GB 或 16GB 显存机器。

### 输出不是严格 JSON

脚本会兜底包装为：

```json
{
  "image_type": "其他",
  "summary": "模型原始输出",
  "uncertain": ["模型输出不是严格 JSON，需要人工检查或二次清洗"]
}
```

如果经常发生，可以把 `--max-new-tokens` 调大，或者在 prompt 中继续强化“只输出 JSON”。

## 7. 和 TinyRAG 集成的下一步

这个 demo 暂时只负责：

```text
图片 -> Qwen2.5-VL 结构化说明 -> RAG 文本块
```

下一步建议再做：

```text
docx/pdf/pptx 自动提取图片
OCR 提取图片中文字
图片说明文本用 BGE embedding 入库
检索命中图片块时返回 image_path
```
