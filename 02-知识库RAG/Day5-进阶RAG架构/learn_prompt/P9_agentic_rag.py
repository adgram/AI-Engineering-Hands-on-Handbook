import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
import json
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p9")

rag = BaseRAG(persist_dir=db_path, collection_name="agentic_rag_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

class AgenticRAG:
    """
    Agentic RAG：LLM 自主决策检索过程

    行动空间：
    1. search(query) — 检索知识库
    2. ask(user_msg) — 向用户追问澄清
    3. answer(final) — 给出最终回答
    """

    def __init__(self, rag, max_steps: int = 6):
        self.collection = rag.store
        self.rag = rag
        self.max_steps = max_steps
        self.history = []

    def _think(self, query: str, context: str) -> dict:
        history_str = "\n".join([
            f"第{h['step']}步: 行动={h['action']}, 结果={h['result'][:100]}"
            for h in self.history[-3:]
        ])

        resp = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"""你是 RAG 智能体。

问题：{query}

已检索到的内容：{context[:500] if context else '无'}

决策历史：
{history_str or '无'}

请决定下一步行动（输出 JSON）：
1. 需要更多信息 → {{"action": "search", "query": "改进后的搜索词"}}
2. 需要向用户确认 → {{"action": "ask", "message": "追问内容"}}
3. 已有足够信息 → {{"action": "answer", "response": "最终回答"}}

输出严格 JSON："""}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)

    def _search(self, query: str, top_k: int = 3) -> str:
        results = self.collection.search(query, n_results=top_k)
        if results['documents']:
            return "\n".join(results['documents'][0][:top_k])
        return "未找到相关信息"

    def run(self, query: str) -> str:
        context = ""
        step = 0

        while step < self.max_steps:
            decision = self._think(query, context)
            action = decision.get("action", "answer")

            self.history.append({
                "step": step + 1,
                "action": action,
                "reason": decision.get("query", decision.get("message", "")),
                "result": "",
            })

            if action == "search":
                result = self._search(decision.get("query", query))
                context += f"\n--- 检索结果 ---\n{result}"
                self.history[-1]["result"] = result[:100]

            elif action == "ask":
                return f"[追问用户] {decision.get('message')}"

            elif action == "answer":
                return decision.get("response", "无法回答")

            step += 1

        return f"已达最大步数，基于已有信息回答：\n{context[:300]}"


class ReflectiveAgenticRAG(AgenticRAG):
    def _evaluate_sufficiency(self, query: str, context: str) -> dict:
        resp = self.client.chat(
            messages=[{"role": "user", "content": f"""评估以下信息是否足够回答问题。

问题：{query}

信息：
{context[:800]}

输出 JSON：
- 足够：{{"sufficient": true, "confidence": 0-10, "draft_answer": "..."}}
- 不足：{{"sufficient": false, "missing": "缺失的信息", "next_query": "改进搜索词"}}"""}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"Agentic RAG已配置，支持search/ask/answer行动")
print(f"结果已写入 {_output_file}")