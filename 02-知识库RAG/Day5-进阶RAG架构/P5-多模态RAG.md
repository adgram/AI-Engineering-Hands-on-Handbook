# P5：多模态 RAG（图片/文字混合检索，可选）

## 目标
了解多模态 RAG 的基本实现方案

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-P1（Embedding 原理：文本→向量） |
| 本章内容 | 把 Embedding 从文本扩展到图文联合空间，介绍文字描述索引、CLIP 多模态 Embedding、统一语义空间和 Colpali 视觉文档检索四种方案。 |

## 什么是多模态 RAG？

传统 RAG 仅支持文本检索，多模态 RAG 扩展为同时检索图片、图表、表格等多种数据格式。

```
传统 RAG：只能检索文本
多模态 RAG：可以检索图片、图表、表格、代码等
```

## 实现方案对比

从最简单的文字描述索引到多模态 Embedding，不同方案在成本、难度和效果上差异显著，需根据场景权衡。

| 方案 | 说明 | 依赖 | 效果 |
|------|------|------|------|
| 文字描述索引 | 用文字描述图片内容，存为文本 | 无特殊依赖 | 一般 |
| 多模态 Embedding | 图片和文字在同一向量空间 | 多模态模型 | 好 |
| 截图 + OCR | 提取图片中的文字 | OCR 工具 | 仅限文字 |

## 方案 1：文字描述索引（最简单）

用 LLM 生成图片的文字描述后存入向量库，检索时通过文字匹配定位图片，实现成本最低。

```python
def index_image_with_description(image_path: str, llm_client) -> str:
    """用 LLM 描述图片内容，将描述存入向量库"""
    import base64
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    # 使用支持图片的模型描述图片（如通过 SiliconFlow 调用的视觉模型）
    response = llm_client.chat(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "详细描述这张图片的内容"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}
        ]
    )
    
    description = response.choices[0].message.content
    
    # 将描述存入文本向量库
    return description

# 这样图片的描述就可以通过文字搜索被检索到了
```

## 方案 2：多模态 Embedding

使用 CLIP 等模型将图片和文本映射到同一向量空间，可直接计算图文相似度，实现真正的跨模态检索。

```python
# 使用 CLIP 等模型进行图-文联合 Embedding
# pip install transformers torch

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPEmbedding:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_text(self, text: str) -> list:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.model.get_text_features(**inputs)
        return outputs.detach().numpy()[0].tolist()
    
    def embed_image(self, image_path: str) -> list:
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()[0].tolist()

# 使用：图片和文字在同一向量空间，可以直接互相检索
clip = CLIPEmbedding()
text_vec = clip.embed_text("一只猫在晒太阳")
# image_vec = clip.embed_image("cat.jpg")
# 余弦相似度计算图文匹配度
```

## 方案 3：统一语义空间（Unified Semantic Space）

使用 Qwen3-VL-embedding 等多模态模型，将文本和图片映射到同一向量空间，实现图文互检：

```
核心流程：
文本/图片 → 同一编码器 → 同一向量空间 → 任意跨模态检索

应用场景：电商搜索"红色连衣裙"→ 同时返回文本描述和商品图片。

参考模型：Qwen3-VL-embedding、CLIP
```

实现要点：
1. 文本和图片通过同一模型编码为相同维度的向量
2. 文本搜索时，query 向量同时与文本索引和图片索引计算余弦相似度
3. 以图搜图同理，图片向量同时匹配文本和图片

## 方案 4：Colpali — 视觉文档检索

Colpali 直接对文档页面进行视觉编码，保留版面信息，特别适合复杂排版的扫描件：

```
核心思路：
文档页面截图 → Colpali 视觉编码器 → 向量索引
文本查询 → 同一编码器 → 与页面向量匹配

应用场景：PDF 扫描件、发票、表单、合同等复杂排版文档

参考项目：vidore/colpali-v1.2（https://github.com/illuin-tech/colpali）
```

对比传统 OCR + 文本检索，Colpali 直接理解页面布局（表格、分栏、标题层级），无需先提取文字。

## 多模态 RAG 方案选择

不同场景适合不同方案：快速原型用文字描述索引，生产级图文互检用多模态 Embedding，复杂排版文档用 Colpali。

| 方案 | 成本 | 实施难度 | 效果 | 适用场景 |
|------|------|---------|------|----------|
| 文字描述索引 | 低 | 低 | 一般 | 快速原型 |
| 多模态 Embedding(CLIP) | 低 | 中 | 好 | 通用图文互检 |
| 统一语义空间 | 中 | 高 | 好 | 生产级多模态 |
| Colpali 视觉编码 | 高 | 高 | 很好 | 扫描件/复杂排版 |
| 截图 + OCR | 低 | 低 | 限文字 | 图中有大量文字 |

## 完成标准
- [ ] 了解多模态 RAG 的实现方案
- [ ] 理解图文互检的基本原理
- [ ] 了解统一语义空间和 Colpali 等前沿方案

## 下一步 → P6-增量更新.md
