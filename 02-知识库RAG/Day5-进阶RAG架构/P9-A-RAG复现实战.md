# P9：A-RAG 复现实战 — 从零构建 Agentic RAG

## 目标
从零实现 Agentic RAG（A-RAG），让 LLM 自主决策何时检索、搜什么、是否追问

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day2-P2（ReAct 循环：推理→行动→观察）+ Day4-P1（NaiveRAG 固定流程）+ Day5-P4（A-RAG 三层检索工具） |
| 本章内容 | 把 ReAct 应用到 RAG，让 LLM 自主决策行动空间（search/ask/answer），突破固定流程 RAG 的局限，应对复杂查询。 |

## Agentic RAG 核心循环

```
用户提问 → LLM 分析意图 → 决策行动(检索/追问/回答) → 观察结果 → 重复 → 最终回答
```

Agentic RAG 的核心是一个 ReAct 循环：LLM 每步分析当前状态，从 search（检索）、ask（追问用户）、answer（最终回答）三种行动中选择一种，执行后观察结果并进入下一轮，直至给出最终答案。

```python
import json

class AgenticRAG:
    """
    Agentic RAG：LLM 自主决策检索过程
    
    行动空间：
    1. search(query) — 检索知识库
    2. ask(user_msg) — 向用户追问澄清
    3. answer(final) — 给出最终回答
    """
    
    def __init__(self, collection, llm_client, max_steps: int = 6):
        self.collection = collection
        self.client = llm_client
        self.max_steps = max_steps
        self.history = []  # 记录决策轨迹
    
    def _think(self, query: str, context: str) -> dict:
        """LLM 分析现状，决定下一步行动"""
        history_str = "\n".join([
            f"第{h['step']}步: 行动={h['action']}, 结果={h['result'][:100]}"
            for h in self.history[-3:]  # 只看最近3步
        ])
        
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
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
        """执行检索"""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if results['documents']:
            return "\n".join(results['documents'][0][:top_k])
        return "未找到相关信息"
    
    def run(self, query: str) -> str:
        """主循环"""
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
                # 改进搜索词后再检索
                result = self._search(decision.get("query", query))
                context += f"\n--- 检索结果 ---\n{result}"
                self.history[-1]["result"] = result[:100]
                
            elif action == "ask":
                # 向用户追问（此处模拟用户回答）
                return f"[追问用户] {decision.get('message')}"
                
            elif action == "answer":
                return decision.get("response", "无法回答")
            
            step += 1
        
        # 达到最大步数，强制总结
        return f"已达最大步数，基于已有信息回答：\n{context[:300]}"

# 使用
# agent = AgenticRAG(collection, client)
# answer = agent.run("RAG 和微调有什么区别？各自适用什么场景？")
# print(answer)
```

## A-RAG 决策轨迹可视化

以下函数将 A-RAG 的决策历史以可视化的方式打印出来，展示每一步的行动类型、决策原因和检索结果，便于调试和理解智能体的行为。

```python
def visualize_trajectory(agent: AgenticRAG, query: str, final_answer: str):
    """可视化 A-RAG 的决策轨迹"""
    print("=" * 60)
    print(f"问题: {query}")
    print("=" * 60)
    
    for i, h in enumerate(agent.history, 1):
        action_icon = {"search": "🔍", "ask": "❓", "answer": "✅"}
        icon = action_icon.get(h["action"], "➡️")
        print(f"\n{icon} 第{i}步: {h['action']}")
        print(f"   原因: {h['reason']}")
        if h["result"]:
            print(f"   结果: {h['result'][:80]}...")
    
    print("\n" + "=" * 60)
    print(f"最终回答: {final_answer[:100]}...")

# 使用
# visualize_trajectory(agent, "RAG 和微调有什么区别？", "结合检索和微调...")
```

## 高级特性：带反思的 A-RAG

在基础 A-RAG 之上增加自我反思机制，每次检索后评估当前信息是否足以回答问题，若充足则提前终止并生成回答，避免不必要的重复检索。

```python
class ReflectiveAgenticRAG(AgenticRAG):
    """带自我反思的 A-RAG：每次检索后评估是否足够回答"""
    
    def _evaluate_sufficiency(self, query: str, context: str) -> dict:
        """评估当前信息是否足够回答"""
        resp = self.client.chat.completions.create(
            model="deepseek-v4-flash",
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
    
    def run(self, query: str) -> str:
        """带反思的主循环：每次检索后评估信息充分性"""
        context = ""
        step = 0
        while step < self.max_steps:
            decision = self._think(query, context)
            action = decision.get("action", "answer")
            self.history.append({"step": step + 1, "action": action, "reason": decision.get("query", ""), "result": ""})
            
            if action == "search":
                result = self._search(decision.get("query", query))
                context += f"\n--- 检索结果 ---\n{result}"
                self.history[-1]["result"] = result[:100]
                # 反思：评估当前信息是否足够
                eval_result = self._evaluate_sufficiency(query, context)
                if eval_result.get("sufficient"):
                    return eval_result.get("draft_answer", "信息充足，但无法生成回答")
            elif action == "ask":
                return f"[追问用户] {decision.get('message')}"
            elif action == "answer":
                return decision.get("response", "无法回答")
            step += 1
        return f"已达最大步数，基于已有信息回答：\n{context[:300]}"

```

## A-RAG vs 传统 RAG

下表从检索次数、搜索策略、信息评估、追问能力等多个维度对比传统 RAG 与 A-RAG 的关键差异。

| 维度 | 传统 RAG | A-RAG |
|------|---------|-------|
| 检索次数 | 固定 1 次 | 动态，可多轮 |
| 搜索策略 | 用户原始查询 | 可改写/拆解 |
| 信息评估 | 无 | 自我反思 |
| 追问能力 | 无 | 可向用户追问 |
| Token 消耗 | 低 | 中-高 |
| 复杂查询 | 效果差 | 效果好 |

## 动手实验

1. 实现基础的 Agentic RAG 循环
2. 测试多轮搜索后回答复杂问题
3. 加入自我反思机制
4. 对比 A-RAG 和传统 RAG 在复杂问题上的表现

## 完成标准
- [ ] 理解 Agentic RAG 的核心循环
- [ ] 实现搜索/追问/回答三种行动
- [ ] 验证 A-RAG 在复杂查询上的优势

## 下一步 → Day6-RAG生产化实践
