# P7：小项目 — 本地文档问答 CLI 工具

## 目标
整合Day3所学，构建一个支持多种文档格式的本地问答工具

## 架构概览

整个工具按职责拆分为 6 个模块，数据流向如下：

```
用户提问 → main.py (CLI 入口)
               ↓
         qa_engine.py (检索 + LLM 生成)
            ↙        ↘
    vector_store.py    config.py
         ↓              ↑
    chromadb ─── embedding(API) ──→ SiliconFlow BAAI/bge-m3
         ↑
    chunker.py ← document_loader.py ← 本地文件 (TXT/MD/PDF)
```

### 职责划分

| 角色 | 负责的模块 | 做什么 |
|------|-----------|--------|
| **向量数据库** | vector_store.py → chromadb | 存储文档向量，执行相似度检索，返回最相关的 K 条文档块 |
| **Embedding API** | SiliconFlow BAAI/bge-m3 | 将文本转化为语义向量（连接"文本世界"和"向量世界"的桥梁） |
| **用户自定义** | document_loader.py / chunker.py / config.py | ① 选择加载的本地文档路径和格式 ② 配置切分粒度（chunk_size / overlap）③ 选择 Embedding 模型 |

> LLM 生成回答部分（qa_engine.py 中的 LLM 调用）属于 Day4 RAG 内容，Day3 聚焦 Embedding + 向量库 + 文档切分。

## 项目结构

```
doc_qa_tool/
├── main.py             # CLI 入口
├── document_loader.py  # 文档加载器（支持 TXT/MD/PDF）
├── chunker.py          # 文档切分器
├── vector_store.py     # 向量存储封装
├── qa_engine.py        # 问答引擎（检索 + 生成）
├── config.py           # 配置
└── .env
```

## 各模块实现

### config.py — 集中管理所有配置

把 API Key、模型名、路径等配置抽离到单独文件，避免硬编码。其他模块通过 `from config import ...` 引用：

```python
import os
from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = "./chroma_db_doc_qa"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"              # 通过 SiliconFlow API 调用
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-v4-flash"
TOP_K = 5
```

### document_loader.py — 支持三种格式的文档加载器

根据文件扩展名自动选择加载方式。支持 .txt / .md / .pdf，可递归遍历目录加载所有文件：

```python
import os

def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_markdown(file_path: str) -> str:
    return load_text(file_path)

def load_pdf(file_path: str) -> str:
    # 需要安装 PyMuPDF: pip install pymupdf
    import fitz
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".txt": load_text, ".md": load_markdown, ".pdf": load_pdf}
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"不支持的文件格式: {ext}")
    return loader(file_path)

def load_directory(dir_path: str) -> list:
    """加载目录下所有支持的文件"""
    supported = [".txt", ".md", ".pdf"]
    docs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported:
                path = os.path.join(root, f)
                try:
                    content = load_document(path)
                    docs.append({"path": path, "content": content})
                    print(f"  load: {path}")
                except Exception as e:
                    print(f"  skip {path}: {e}")
    return docs
```

### chunker.py — 文档切分器

基于 LangChain 的 `RecursiveCharacterTextSplitter`（详见 P3-文档切分策略），按中文标点优先级逐级切分，并将结果封装为带来源路径的结构化块：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len
        )

    def chunk_documents(self, docs: list) -> list:
        all_chunks = []
        for doc in docs:
            chunks = self.splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({"file": doc["path"], "chunk_index": i, "text": chunk})
        return all_chunks
```

### vector_store.py — ChromaDB 封装

将 ChromaDB 的集合操作（`PersistentClient` / `get_or_create_collection` / `add` / `query`，详见 P2）封装为简单的业务接口，并支持分批添加避免内存溢出：

```python
import chromadb

class VectorStore:
    def __init__(self, persist_dir="./chroma_db", embedding_fn=None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="doc_qa", embedding_function=embedding_fn
        )

    def add_chunks(self, chunks: list):
        texts = [c["text"] for c in chunks]
        ids = [f"{c['file']}#{c['chunk_index']}" for c in chunks]
        metadatas = [{"file": c["file"], "index": c["chunk_index"]} for c in chunks]
        for i in range(0, len(texts), 100):
            self.collection.add(documents=texts[i:i+100], ids=ids[i:i+100], metadatas=metadatas[i:i+100])
        print(f"added {len(chunks)} chunks")

    def search(self, query: str, n_results: int = 5, where_filter: dict = None):
        kwargs = {"query_texts": [query], "n_results": n_results}
        if where_filter:
            kwargs["where"] = where_filter
        return self.collection.query(**kwargs)

    def count(self):
        return self.collection.count()
```

### qa_engine.py — 检索 + 生成

核心问答引擎。先调用 `vector_store.search` 检索相关文档块，拼接为上下文后发给 LLM 生成回答，并标注引用来源：

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class QAEngine:
    def __init__(self, vector_store, llm_client=None):
        self.vector_store = vector_store
        self.client = llm_client or OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

    def ask(self, question: str, n_results: int = 5) -> dict:
        results = self.vector_store.search(question, n_results=n_results)

        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}

        context_parts = []
        sources = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context_parts.append(f"[来源 {i+1}] {doc}")
            sources.append({"file": meta['file'], "chunk": doc[:100]})

        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": "你是知识库问答助手，基于提供的参考资料回答问题。如果信息不足，请如实说明。引用来源时标注 [来源 N]。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources
        }
```

### main.py — CLI 入口

支持三种运行模式：`--init` 初始化知识库、`--ask` 单次问答、无参数进入交互模式。Embedding 函数在这里定义并注入，避免模块间耦合：

```python
import argparse, os, sys
from pathlib import Path

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))

from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
from document_loader import load_directory
from chunker import Chunker
from vector_store import VectorStore
from qa_engine import QAEngine
from config import CHROMA_PATH, EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL

class _EF(EmbeddingFunction):
    def __init__(self):
        self.client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    def __call__(self, input: Documents) -> list:
        texts = input if isinstance(input, list) else [input]
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

def main():
    parser = argparse.ArgumentParser(description="本地文档问答工具")
    parser.add_argument("--init", help="初始化知识库，指定文档目录路径")
    parser.add_argument("--ask", help="单次问答，输入问题")
    parser.add_argument("--persist", default=CHROMA_PATH, help="向量数据库持久化路径")
    args = parser.parse_args()

    vs = VectorStore(persist_dir=args.persist, embedding_fn=_EF())
    qa = QAEngine(vs)

    if args.init:
        print(f"正在加载: {args.init}")
        docs = load_directory(args.init)
        chunks = Chunker().chunk_documents(docs)
        vs.add_chunks(chunks)
        print(f"知识库就绪，共 {vs.count()} 个文档块")

    elif args.ask:
        result = qa.ask(args.ask)
        print(f"\n答案: {result['answer']}")
        if result['sources']:
            print(f"\n来源 ({len(result['sources'])}):")
            for s in result['sources']:
                print(f"  文件: {s['file']}")

    else:
        print("本地文档问答工具（输入 quit 退出）")
        print(f"知识库文档块数: {vs.count()}")
        while True:
            q = input("\n问题: ").strip()
            if q.lower() in ("quit", "exit", "q", "退出"):
                break
            result = qa.ask(q)
            print(f"\n{result['answer']}")

if __name__ == "__main__":
    main()
```

## 使用方式

```bash
# 1. 初始化知识库
python main.py --init ./my_documents/

# 2. 单次问答
python main.py --ask "RAG 的工作原理是什么？"

# 3. 交互模式
python main.py
```

## 设计要点

- **依赖注入**：`VectorStore` 接受 `embedding_fn` 参数，可换用不同模型；`QAEngine` 接受 `llm_client` 参数，可换用不同 LLM
- **ID 编码**：文档块 ID 编码了来源路径和索引号（`file#index`），方便溯源
- **分批添加**：避免一次性添加大量文档导致 ChromaDB 内存溢出
- **Embedding 函数复用**：使用了 P4/P5/P6 相同的 `_EF` 模式，通过 SiliconFlow API 调用 BAAI/bge-m3

## 验收清单

- [ ] 支持 TXT / MD / PDF 三种格式
- [ ] 文档切分 + 向量化存储
- [ ] 检索 + LLM 生成回答
- [ ] 回答中标注信息来源
- [ ] 交互式问答和命令行两种模式
- [ ] 支持重新初始化知识库

## 下一步 → Day4 RAG检索增强生成
