import argparse, sys
from typing import Optional
from pathlib import Path

_base = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(_base))

from qa_engine import QAEngine
from config import CHROMA_PATH, EMBEDDING_API_KEY, EMBEDDING_BASE_URL
from common.rag_base import SiliconFlowEmbedding, Chunker, load_directory, EmbeddingFunction
from common.rag_base import VectorStore as _VS

class VectorStore(_VS):
    def __init__(self, persist_dir: str, collection_name: str = "knowledge",
                 embedding_fn: Optional[EmbeddingFunction] = None):
        super().__init__(persist_dir, collection_name, embedding_fn)
        self._ingested_ids = set(self.collection.get()["ids"])  # 已录入 ID 集合（防重）

    def add_chunks(self, chunks: list):
        texts = [c["text"] for c in chunks]
        ids = [f"{c['file']}#{c['chunk_index']}" for c in chunks]
        new_indices = [i for i, cid in enumerate(ids) if cid not in self._ingested_ids]
        if not new_indices:
            return
        metadatas = [{"file": c["file"], "index": c["chunk_index"]} for c in chunks]
        self.add(texts, ids, metadatas)
        print(f"已添加 {len(chunks)} 个文档块")

def main():
    parser = argparse.ArgumentParser(description="本地文档问答工具")
    parser.add_argument("--init", help="初始化知识库，指定文档目录路径")
    parser.add_argument("--ask", help="单次问答，输入问题")
    parser.add_argument("--persist", default=CHROMA_PATH, help="向量数据库持久化路径")
    args = parser.parse_args()

    embedding_fn = SiliconFlowEmbedding()
    embedding_fn.client.api_key = EMBEDDING_API_KEY
    embedding_fn.client.base_url = EMBEDDING_BASE_URL
    vs = VectorStore(persist_dir=args.persist, embedding_fn=embedding_fn)
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
