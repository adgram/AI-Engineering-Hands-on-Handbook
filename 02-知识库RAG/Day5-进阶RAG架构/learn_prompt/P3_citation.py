import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p3")

rag = BaseRAG(persist_dir=db_path, collection_name="citation_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

def generate_with_citation(query: str, docs_with_meta: list) -> str:
    """让 LLM 生成带引用的回答"""
    
    # 构建带编号的上下文
    context_parts = []
    for i, (doc, meta) in enumerate(docs_with_meta):
        source = meta.get('source', meta.get('file', f'来源{i+1}'))
        context_parts.append(f"[{i+1}] (来源: {source})\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    messages = [
        {"role": "system", "content": f"""你是一个知识库问答助手。回答规则：
1. 基于参考资料回答
2. 在引用信息的句末标注来源编号，如[1]、[2]
3. 如果引用了多个来源，标注多个编号，如[1][2]
4. 没有资料来源的内容不要编造
5. 格式：[编号](来源名称)"""},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"}
    ]
    
    response = rag.llm.chat.completions.create(
        model=rag.llm_model,
        messages=messages,
    )
    return response.choices[0].message.content

# === Code Block 2 ===

def structured_citation(query: str, docs_with_meta: list) -> dict:
    """输出结构化的带引用回答"""
    
    context_parts = []
    for i, (doc, meta) in enumerate(docs_with_meta):
        source = meta.get('file', meta.get('source', f'来源{i+1}'))
        context_parts.append(f"[{i+1}] {doc}")
    
    context = "\n\n".join(context_parts)
    
    prompt = (
        f"参考资料：\n{context}\n\n问题：{query}\n\n"
        '输出 JSON：\n'
        '{"answer": "你的回答", "citations": [{"number": 1, "text": "引用的具体句子", "source": "来源文件名"}]}'
    )
    response = rag.llm.chat.completions.create(
        model=rag.llm_model,
        messages=[
            {"role": "system", "content": "输出 JSON 格式的回答，包含 answer 和 citations 字段。"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    
    return json.loads(response.choices[0].message.content)

# 使用
results = rag.store.search("RAG 是什么？", n_results=2)
docs_with_meta = list(zip(results['documents'][0], results['metadatas'][0]))

result = structured_citation("RAG 是什么？", docs_with_meta)
print(result['answer'])
for c in result.get('citations', []):
    print(f"  [{c['number']}] {c['text'][:50]}... → {c['source']}")

# === Code Block 3 ===

def posthoc_citation(answer: str, docs: list) -> list:
    """在生成回答后，匹配句子和来源"""
    citations = []
    
    for doc in docs:
        resp = rag.llm.chat.completions.create(
            model=rag.llm_model,
            messages=[{
                "role": "user",
                "content": f"""判断以下回答中的哪些内容来自这篇资料。

回答：{answer}

资料：{doc}

输出回答中和资料匹配的句子，每句一行。如果不匹配，输出"无"。"""
            }],
        )
        matched = resp.choices[0].message.content
        if matched and matched != "无":
            citations.append({"source": doc[:80], "matched_sentences": matched})
    
    return citations

# === Code Block 4 ===

def format_answer_with_citations(rag_result: dict) -> str:
    """格式化带引用的回答"""
    answer = rag_result['answer']
    sources = rag_result.get('sources', [])
    
    lines = [answer, "\n---\n来源:"]
    for i, src in enumerate(sources, 1):
        file_name = src.get('file', src.get('source', '未知'))
        excerpt = src.get('content', str(src))[:100]
        lines.append(f"  [{i}] {file_name}")
        lines.append(f"      {excerpt}...")
    
    return "\n".join(lines)

# 使用
result = rag.query("RAG 的优势是什么？")
print(format_answer_with_citations(result))

# === Code Block 5 ===

def evaluate_citation_quality(answer: str, contexts: list) -> dict:
    """评估引用质量"""
    
    prompt = f"""评估以下回答的引用质量。

回答：{answer}

资料：
{"".join([f"[{i+1}] {c}\n" for i, c in enumerate(contexts)])}

评估标准：
1. 引用准确率：引用是否真实来自资料？
2. 引用覆盖率：关键信息是否有引用？
3. 幻觉率：是否有未引用的编造内容？

输出 JSON：
{{
    "citation_accuracy": 0-10,
    "citation_coverage": 0-10,
    "hallucination": true/false,
    "reason": "..." 
}}"""
    
    resp = rag.llm.chat.completions.create(
        model=rag.llm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    return json.loads(resp.choices[0].message.content)


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"结构化引用结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
print(f"结果已写入 {_output_file}")