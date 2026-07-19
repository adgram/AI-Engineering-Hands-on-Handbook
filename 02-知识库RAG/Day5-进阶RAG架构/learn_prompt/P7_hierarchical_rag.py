import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
from typing import Literal
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p7")

rag = BaseRAG(persist_dir=db_path, collection_name="hierarchical_rag_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

class HierarchicalRAG:
    """
    分层 RAG：多级知识库路由检索

    层级设计：
    1. 全局层（Global）：高频/热门知识，小索引，低延迟
    2. 领域层（Domain）：按业务领域划分的中粒度知识
    3. 文档层（Document）：完整文档库，大索引，高延迟
    """

    def __init__(self, global_collection, domain_collections: dict, doc_collection, rag):
        self.global_col = global_collection
        self.domain_cols = domain_collections
        self.doc_col = doc_collection
        self.rag = rag

    def route(self, query: str) -> Literal["global", "domain", "document", "ask_llm"]:
        resp = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""判断以下查询最适合哪个知识库层级：
- global：基础概念、通用知识
- domain：需要特定领域专业知识
- document：需要查阅具体文档细节
- ask_llm：不需要检索，LLM 可直接回答

查询：{query}

只输出一个词。"""}],
            temperature=0.0,
        )
        level = resp.choices[0].message.content.strip().lower()
        return level if level in ("global", "domain", "document", "ask_llm") else "document"

    def route_domain(self, query: str) -> str:
        topics = list(self.domain_cols.keys())
        resp = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""可用领域：{', '.join(topics)}
查询：{query}
最适合的领域是？只输出领域名。"""}],
            temperature=0.0,
        )
        domain = resp.choices[0].message.content.strip()
        return domain if domain in self.domain_cols else topics[0]

    def retrieve(self, query: str, top_k: int = 5) -> list:
        level = self.route(query)

        if level == "ask_llm":
            return []

        if level == "global":
            results = self.global_col.query(query_texts=[query], n_results=top_k)
        elif level == "domain":
            domain = self.route_domain(query)
            results = self.domain_cols[domain].query(query_texts=[query], n_results=top_k)
        else:
            results = self.doc_col.query(query_texts=[query], n_results=top_k)

        return results['documents'][0] if results['documents'] else []


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"分层RAG已配置，层级: global + domain + document")
print(f"结果已写入 {_output_file}")