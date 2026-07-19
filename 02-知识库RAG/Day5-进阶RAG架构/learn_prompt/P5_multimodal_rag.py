import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from common.llm_client import LLMClient
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p5")

rag = BaseRAG(persist_dir=db_path, collection_name="multimodal_rag_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

# 创建支持图片的多模态客户端（使用 SiliconFlow 托管的视觉模型）
vision_client = LLMClient(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3-VL-8B-Instruct",
)

def index_image_with_description(image_path: str, llm_client=None) -> str:
    """用多模态 LLM 描述图片内容，返回文本描述"""
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    client = llm_client or vision_client
    response = client.chat(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "详细描述这张图片的内容"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}
        ]
    )

    description = response.choices[0].message.content
    return description

# 使用示例（取消注释并指定图片路径即可运行）：
# desc = index_image_with_description("path/to/your/image.png")
# rag.add_documents(documents=[desc], ids=["img_desc_1"])
# print(f"图片描述已入库: {desc[:100]}...")

# === Code Block 2 ===

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

# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"CLIP文本嵌入完成，向量维度: {len(text_vec)}")
print(f"结果已写入 {_output_file}")