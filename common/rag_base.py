"""RAG 基类 — 提供中文 Embedding、向量存储、文档切分、检索→增强→生成完整链路"""

import os, hashlib
from typing import Optional
from chromadb import Documents, EmbeddingFunction, PersistentClient, ClientAPI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SiliconFlowEmbedding(EmbeddingFunction):
    """支持中文的 Embedding 函数，使用 BAAI/bge-m3 模型（通过 SiliconFlow API）"""
    def __init__(self, model: str = "BAAI/bge-m3"):
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY 未设置，RAG 章节需要此 Key 运行 Embedding 模型")
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")  # SiliconFlow 客户端
        self.model = model                                                               # Embedding 模型名称

    def __call__(self, input: Documents) -> list[list[float]]:
        texts = input if isinstance(input, list) else [input]
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [data.embedding for data in response.data]


class VectorStore:
    """ChromaDB 向量存储封装：自动建库、批量写入、语义检索"""
    def __init__(self, persist_dir: str, collection_name: str = "knowledge",
                 embedding_fn: Optional[EmbeddingFunction] = None):
        self.client:ClientAPI = PersistentClient(path=persist_dir)                       # ChromaDB 持久化客户端
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn                        # 集合（自动建库）
        )

    def add(self, texts: list[str], ids: list[str],
            metadatas: Optional[list[dict]] = None, batch_size: int = 100):
        """批量写入，支持按批次提交"""
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            self.collection.add(
                documents=batch,
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size] if metadatas else None
            )

    def search(self, query: str, n_results: int = 5,
               where_filter: Optional[dict] = None) -> dict:
        """语义检索，支持按 metadata 过滤"""
        kwargs = {"query_texts": [query], "n_results": n_results}
        if where_filter:
            kwargs["where"] = where_filter
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()


class Chunker:
    """文档切分器：按递归字符切分，支持中文标点"""
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(                                 # 递归字符文本切分器
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )

    def chunk_text(self, file_path: str, text: str) -> list[dict]:
        chunks = self.splitter.split_text(text)
        return [
            {"file": file_path, "chunk_index": i, "text": chunk}
            for i, chunk in enumerate(chunks)
        ]

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_text(doc["path"], doc["content"]))
        return all_chunks


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    import fitz
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_document(file_path: str) -> str:
    """按文件扩展名自动选择加载器（.txt/.md/.pdf）"""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".txt": load_text, ".md": load_text, ".pdf": load_pdf}
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"不支持的文件格式: {ext}")
    return loader(file_path)


def load_directory(dir_path: str) -> list[dict]:
    """递归加载目录下所有支持格式的文档"""
    supported = {".txt", ".md", ".pdf"}
    docs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported:
                path = os.path.join(root, f)
                try:
                    content = load_document(path)
                    docs.append({"path": path, "content": content})
                    print(f"  loaded: {path}")
                except Exception as e:
                    print(f"  skip {path}: {e}")
    return docs


class BaseRAG:
    """RAG 基类：Embedding + 向量库 + LLM 一站式封装

    - 初始化时自动创建 VectorStore（含中文 Embedding）
    - 提供 add_documents / add_chunks 写入
    - query() 执行完整 检索→增强→生成 流程
    - 内置结果缓存，避免重复查询
    """
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "knowledge",
        embedding_fn: Optional[EmbeddingFunction] = None,
        llm_model: str = "deepseek-v4-flash",
    ):
        self.llm = OpenAI(                              # OpenAI LLM 客户端（生成回答用）
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.llm_model = llm_model                      # 生成模型名称
        self.store = VectorStore(                       # 向量存储实例
            persist_dir, collection_name,
            embedding_fn or SiliconFlowEmbedding()
        )
        self.cache = {}                                     # 查询结果缓存 {question: result}
        self._ingested_ids = set(self.store.collection.get()["ids"])  # 已录入 ID 集合（防重）
        self.stats = {"total_calls": 0, "total_tokens": 0}  # 统计信息

    def add_documents(self, documents: list[str],
                      metadatas: Optional[list[dict]] = None,
                      ids: Optional[list[str]] = None) -> int:
        '''传入文本片段'''
        if ids is None:
            ids = [hashlib.md5(d.encode()).hexdigest()[:16] for d in documents]
        new_ids = [i for i in ids if i not in self._ingested_ids]
        if not new_ids:
            return 0
        new_docs = [documents[ids.index(i)] for i in new_ids]
        new_metas = [metadatas[ids.index(i)] for i in new_ids] if metadatas else None
        self.store.add(new_docs, new_ids, new_metas)
        self._ingested_ids.update(new_ids)
        return len(new_ids)

    def add_chunks(self, chunks: list[dict]):
        '''传入{file, chunk_index, text}格式带路径片段'''
        texts = [c["text"] for c in chunks]
        ids = [f"{c['file']}#{c['chunk_index']}" for c in chunks]
        new_indices = [i for i, cid in enumerate(ids) if cid not in self._ingested_ids]
        if not new_indices:
            return
        new_texts = [texts[i] for i in new_indices]
        new_ids = [ids[i] for i in new_indices]
        new_metadatas = [{"file": chunks[i]["file"], "index": chunks[i]["chunk_index"]} for i in new_indices]
        self.store.add(new_texts, new_ids, new_metadatas)
        self._ingested_ids.update(new_ids)

    def query(self, question: str, k: int = 5, use_cache: bool = True) -> dict:
        """检索 → 拼接上下文 → LLM 生成 → 返回 answer + sources"""
        if use_cache and question in self.cache:
            self.stats["total_calls"] += 1
            return self.cache[question]

        results = self.store.search(question, n_results=k)
        if not results["documents"][0]:
            return {"answer": "未找到相关信息", "sources": []}

        context_parts = []
        sources = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            context_parts.append(f"[source {i+1}] {doc}")
            sources.append({"file": meta.get("file", ""), "content": doc[:200]})

        context = "\n\n".join(context_parts)
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是企业知识库助手，基于资料回答问题并标注来源。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
            ],
        )

        answer = response.choices[0].message.content
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += response.usage.total_tokens

        result = {
            "answer": answer,
            "sources": sources,
            "tokens": response.usage.total_tokens,
        }
        if use_cache:
            self.cache[question] = result
        return result

    def get_stats(self) -> dict:
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "doc_count": self.store.count(),
        }

    def clear_cache(self):
        self.cache.clear()
