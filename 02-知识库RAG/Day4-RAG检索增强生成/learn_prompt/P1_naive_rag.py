"""P1: Naive RAG 完整链路 — 检索 → 增强 → 生成"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()


class NaiveRAG(BaseRAG):
    """Naive RAG 三步流程"""
    def retrieve(self, query: str, k: int = 3) -> dict:
        return self.store.search(query, n_results=k)

    def augment(self, query: str, retrieved_docs: dict) -> list:
        context = "\n\n".join([
            f"[文档{i+1}] {doc}"
            for i, doc in enumerate(retrieved_docs['documents'][0])
        ])

        messages = [
            # 角色设定方法详见 Prompt D1-P2（角色设定完整专题）
            {"role": "system", "content": "你是一个知识库问答助手。请基于以下参考资料回答问题。如果资料不充分，请说'根据现有资料无法完整回答'，并说明缺少哪些信息。"},
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
        ]
        return messages

    def generate(self, messages: list) -> str:
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
        )
        return response.choices[0].message.content

    def ask(self, query: str, k: int = 3) -> dict:
        retrieved = self.retrieve(query, k)
        messages = self.augment(query, retrieved)
        answer = self.generate(messages)

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved['documents'][0],
            "source_metadatas": retrieved['metadatas'][0],
            "distances": retrieved['distances'][0]
        }


if __name__ == "__main__":
    db_path = Path(__file__).parent / "chroma_db_naive"

    rag = NaiveRAG(
        persist_dir=str(db_path),
        collection_name="my_knowledge_base",
    )

    if rag.store.count() == 0:
        from common.rag_base import load_directory
        data_dir = str(Path(__file__).parent.parent.parent.parent / "common" / "text_data" / "rag_knowledge")
        loaded = load_directory(data_dir)
        docs = [d["content"] for d in loaded]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        metas = [{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded]
        
        rag.add_documents(
            documents=docs,
            metadatas=metas,
            ids=doc_ids
        )
        print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

    query = "RAG 相比微调有什么优势？"

    response = rag.llm.chat.completions.create(
        model=rag.llm_model,
        messages=[{"role": "user", "content": query}]
    )
    result = rag.ask(query, k=3)

    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
    with open(_output_file, "w", encoding="utf-8") as _f:
        _f.write(f"查询: {query}\n\n")
        _f.write("=== 纯 LLM 回答 ===\n")
        _f.write(f"{response.choices[0].message.content}\n\n")
        _f.write("=== RAG 回答 ===\n")
        _f.write(f"{result.get('answer', '')}\n\n")
        _f.write("=== 检索到的资料 ===\n")
        for i, (doc, meta) in enumerate(zip(result["sources"], result["source_metadatas"])):
            _f.write(f"来源 {i+1} [{meta.get('topic', 'N/A')}]: {doc}\n")
            _f.write(f"  距离: {result['distances'][i]:.4f}\n")
    print(f"\n结果已写入 {_output_file}")