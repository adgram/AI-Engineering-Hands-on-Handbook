"""P2: Prompt 中注入上下文的最佳实践"""

import json, sys
from pathlib import Path
_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_DIR))
from common.rag_base import BaseRAG
from dotenv import load_dotenv

load_dotenv()


def sort_by_relevance(docs, distances):
    """根据距离（越小越相关）对文档排序"""
    paired = sorted(zip(docs, distances), key=lambda x: x[1])
    return [p[0] for p in paired]


def _ask(llm, prompt: str, model: str = "deepseek-v4-flash") -> str:
    """调用 LLM 获取回答"""
    resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def naive_prompt(docs: list, query: str, llm=None) -> str:
    """简单拼接：将所有文档直接拼接到 prompt 中"""
    context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(docs)])
    prompt = f"参考资料：\n{context}\n\n问题：{query}"
    return _ask(llm, prompt)


def sorted_prompt(docs: list, query: str, llm=None) -> str:
    """排序拼接：按相关性排序后再拼入 prompt"""
    mock_distances = [i for i in range(len(docs))]  # 正序，模拟排序效果
    sorted_docs = sort_by_relevance(docs, mock_distances)
    context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(sorted_docs)])
    prompt = f"参考资料（按相关性排列）：\n{context}\n\n问题：{query}"
    return _ask(llm, prompt)


def map_reduce_rag(docs: list, query: str, llm, model: str = "deepseek-v4-flash") -> str:
    """Map-Reduce 策略：先分别回答每篇文档，再综合生成最终答案"""
    # Map：对每篇文档独立调用 LLM
    partial_answers = []
    for doc in docs:
        resp = llm.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"基于以下资料回答问题。如果该资料不包含答案，返回'无相关信息'。\n\n资料：{doc}\n\n问题：{query}"
            }],
        )
        partial_answers.append(resp.choices[0].message.content)

    # Reduce：将各子回答合并，让 LLM 综合产出最终答案
    all_answers = "\n".join([f"[资料{i+1}] {a}" for i, a in enumerate(partial_answers)])

    final_resp = llm.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"以下是对同一个问题的多个独立回答，请综合它们生成一个完整、一致的最终回答：\n\n{all_answers}\n\n问题：{query}"
        }],
    )

    return final_resp.choices[0].message.content


def compare_injection_strategies(query: str, docs: list, llm):
    """对比三种上下文注入策略的效果"""
    strategies = {
        "简单拼接": lambda: naive_prompt(docs, query, llm),
        "按相关性排列": lambda: sorted_prompt(docs, query, llm),
        "Map-Reduce": lambda: map_reduce_rag(docs, query, llm),
    }

    results = {}
    for name, fn in strategies.items():
        answer = fn()
        results[name] = answer
        print(f"\n=== {name} ===")
        print(answer[:200] + "...")

    return results


if __name__ == "__main__":
    # 初始化 RAG 引擎
    rag = BaseRAG(
        persist_dir=str(Path(__file__).parent / "chroma_db_injection"),
        collection_name="knowledge",
    )

    # 从 rag_knowledge 目录加载文档作为测试数据
    from common.rag_base import load_directory
    data_dir = str(_DIR / "common/text_data/rag_knowledge")
    loaded = load_directory(data_dir)
    docs = [d["content"] for d in loaded]
    print(f"已加载 {len(docs)} 篇文档")

    results = compare_injection_strategies("RAG 相比微调有什么优势？", docs, rag.llm)

    # 将结果写入文件以便后续分析
    _output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.json"
    with open(_output_file, "w", encoding="utf-8") as _f:
        json.dump(results, _f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {_output_file}")