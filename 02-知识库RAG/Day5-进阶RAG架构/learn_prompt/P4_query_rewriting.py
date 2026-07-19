import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p4")

rag = BaseRAG(persist_dir=db_path, collection_name="query_rewriting_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

class QueryRewriter:
    def __init__(self, rag):
        self.rag = rag
    
    def rewrite(self, query: str, context: str = None) -> str:
        """将查询改写为更规范的表达"""
        
        prompt = f"""你是一个查询优化助手。将用户的非正式查询改写为清晰、规范的搜索查询。

规则：
1. 补充术语全称（如 RAG→检索增强生成）
2. 消除口语化表达
3. 纠正错别字
4. 保持原意不变
5. 直接输出改写结果，不要解释

用户查询：{query}
改写后："""
        
        if context:
            prompt = f"对话上下文：{context}\n" + prompt
        
        response = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content.strip()
    
    def expand_with_synonyms(self, query: str) -> list:
        """生成多个查询变体"""
        response = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""为用户查询生成 3 个同义但表述不同的搜索查询，覆盖不同的用词和表达角度。

用户查询：{query}

每行一个查询，直接输出："""}],
        )
        
        variants = response.choices[0].message.content.strip().split("\n")
        return [query] + [v.strip() for v in variants if v.strip()]

# 使用
rewriter = QueryRewriter(rag)
print(rewriter.rewrite("LLM 咋微调啊？"))
print(rewriter.rewrite("RAG 和 fine-tuning 哪个好？"))

# 多查询扩展
variants = rewriter.expand_with_synonyms("向量数据库的优点")
for v in variants:
    print(f"  - {v}")

# === Code Block 2 ===

def multi_query_search(collection, query: str, rewriter, k_per_query: int = 3) -> list:
    """用多个查询变体分别检索，合并结果"""
    queries = rewriter.expand_with_synonyms(query)
    
    all_results = {}
    for q in queries:
        results = collection.query(query_texts=[q], n_results=k_per_query)
        for doc_id, doc, dist in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
            if doc_id not in all_results:
                all_results[doc_id] = {"doc": doc, "dist": dist, "count": 1}
            else:
                all_results[doc_id]["dist"] = min(all_results[doc_id]["dist"], dist)
                all_results[doc_id]["count"] += 1
    
    # 按距离排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["dist"])
    return sorted_results[:k_per_query]

# === Code Block 3 ===

def decompose_query(query: str) -> list:
    """将复杂问题分解为多个子问题"""
    
    response = rag.llm.chat.completions.create(
        model=rag.llm_model,
        messages=[{"role": "user", "content": f"""将以下复杂问题分解为 2-5 个简单子问题。每个子问题应该可以直接用于检索。

复杂问题：{query}

输出格式：每行一个子问题，不要编号以外的内容。

示例：
复杂问题：Python 和 JavaScript 的区别是什么？哪个更容易学？
子问题1: Python 编程语言的特点是什么？
子问题2: JavaScript 编程语言的特点是什么？
子问题3: Python 和 JavaScript 的主要区别
子问题4: Python 和 JavaScript 哪个更容易入门

复杂问题：{query}
子问题1:"""}],
    )
    
    text = response.choices[0].message.content
    sub_queries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("子问题")):
            # 去掉 "子问题1:" 或 "1." 前缀
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            elif ". " in line:
                line = line.split(". ", 1)[1].strip()
            sub_queries.append(line)
    
    return sub_queries

# 使用
queries = decompose_query("RAG 和 Prompt Engineering 有什么区别？各自适合什么场景？")
for i, q in enumerate(queries, 1):
    print(f"子问题{i}: {q}")

# === Code Block 4 ===

class DecompositionRAG:
    def __init__(self, rag):
        self.rag = rag
        self.collection = rag.store
    
    def ask(self, query: str) -> str:
        # 分解
        sub_queries = decompose_query(query)
        print(f"分解为 {len(sub_queries)} 个子问题")
        
        # 对每个子问题检索
        all_contexts = []
        for sq in sub_queries:
            results = self.collection.search(sq, n_results=2)
            for doc in results['documents'][0]:
                if doc not in all_contexts:
                    all_contexts.append(doc)
        
        print(f"共检索到 {len(all_contexts)} 个相关文档块")
        
        # 合并生成
        context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(all_contexts)])
        
        response = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""请基于以下参考资料，全面回答用户的问题。

参考资料：
{context}

问题：{query}

请确保覆盖问题的所有方面。"""}],
        )
        
        return response.choices[0].message.content


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"查询变体数量: {len(variants)}\n原始查询: {variants[0] if variants else '无'}\n子问题数量: {len(queries)}")
print(f"结果已写入 {_output_file}")