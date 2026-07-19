# P2：多轮对话中的 RAG

## 目标
实现支持多轮对话的知识库问答系统

> **前置知识**：
> - 多轮对话的消息结构基础（system/user/assistant 交替追加）见 Prompt D1-P2
> - 滑动窗口原理见 Prompt D2-P1（策略1：滑动窗口）

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day1-P2（消息结构 system/user/assistant 交替）+ Day2-P1（滑动窗口）+ Day4-P1（NaiveRAG 单轮） |
| 本章内容 | 用 LLM 做指代消解（把"它"改写为"RAG"），用滑动窗口管理历史轮次，构建 `ConversationalRAG` 类，让 RAG 从一问一答升级为多轮对话。 |

## 多轮 RAG 的挑战

多轮对话中用户的后续问题往往依赖前文（如代词"它"），直接按单轮方式检索会丢失上下文信息。

```
用户：RAG 是什么？
AI：RAG 是检索增强生成...
用户：它有什么优点？
   → 这里的"它"指 RAG，需要结合历史理解
```

## 挑战 1：指代消解

借助 LLM 将当前问题与历史对话结合，改写为不依赖上下文的独立问题，从而正确理解代词指代对象。

```python
def resolve_query_with_history(query: str, history: list) -> str:
    """用 LLM 把当前问题和历史结合，得到一个独立的问题"""
    if not history:
        return query
    
    history_text = "\n".join([
        f"用户: {h['user']}\nAI: {h['assistant']}"
        for h in history[-3:]  # 只用最近 3 轮
    ])
    
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{
            "role": "user",
            "content": f"""对话历史：
{history_text}

当前问题：{query}

请将当前问题改写为一个独立的问题，不依赖上下文。
直接输出改写后的问题，不要多余内容。"""
        }],
    )
    
    return resp.choices[0].message.content.strip()

# 测试
history = [
    {"user": "什么是 RAG？", "assistant": "RAG 是检索增强生成..."},
    {"user": "它有什么优点？", "assistant": "优点包括减少幻觉..."},
]

rewritten = resolve_query_with_history("它的工作原理是什么？", history)
print(f"原问题: 它的工作原理是什么？")
print(f"改写后: {rewritten}")
```

## 挑战 2：历史上下文注入

在每次生成回答时，将最近的若干轮对话历史拼入 Prompt，让 LLM 感知对话上下文从而给出连贯回复。

```python
class ConversationalRAG:
    def __init__(self, collection, max_history=3):
        self.collection = collection
        self.max_history = max_history
        self.history = []
    
    def ask(self, query: str) -> str:
        # 改写查询（如果需要）
        if self.history:
            standalone_query = resolve_query_with_history(query, self.history)
        else:
            standalone_query = query
        
        # 检索
        results = self.collection.query(
            query_texts=[standalone_query],
            n_results=3
        )
        contexts = results['documents'][0]
        
        # 构建带历史的 Prompt
        history_text = self._format_history()
        context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
        
        messages = [
            {"role": "system", "content": "你是一个知识库问答助手，基于资料和对话历史回答问题。"},
            {"role": "user", "content": f"""对话历史：
{history_text}

参考资料：
{context_text}

新问题：{query}

请结合对话历史和参考资料回答。"""}
        ]
        
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
        )
        
        answer = response.choices[0].message.content
        
        # 保存历史
        self.history.append({"user": query, "assistant": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)  # 滑动窗口的完整实现和变体（摘要压缩/向量检索）见 Prompt D2-P1
        
        return answer
    
    def _format_history(self) -> str:
        lines = []
        for h in self.history[-self.max_history:]:
            lines.append(f"用户: {h['user']}")
            lines.append(f"助手: {h['assistant']}")
        return "\n".join(lines)

# 测试多轮对话
rag = ConversationalRAG(collection)

print(rag.ask("RAG 是什么？"))
print("\n--- 第二轮 ---")
print(rag.ask("它有什么优势？"))
print("\n--- 第三轮 ---")
print(rag.ask("和微调相比呢？"))
```

## 挑战 3：对话状态管理

> 完整的记忆管理方案（短期+长期+三层架构）见 Agent D7-P4，本节仅展示 RAG 场景的状态追踪雏形

```python
class ConversationState:
    """管理对话状态和临时信息"""
    
    def __init__(self):
        self.current_topic = None
        self.mentioned_entities = []
        self.pending_clarification = None
    
    def update(self, query: str, answer: str, retrieved_docs: list):
        """更新对话状态"""
        # 提取当前主题
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": f"从以下对话中提取当前主题词（1-3个词）：\n用户：{query}\nAI：{answer}"}],
        )
        topic = resp.choices[0].message.content.strip()
        if topic and topic != "无":
            self.current_topic = topic
```

## 带历史的多轮 RAG 策略对比

下表对比了查询改写、历史注入、两者结合及完整状态管理四种策略在复杂度、效果和 Token 成本上的差异。

| 策略 | 实现复杂度 | 效果 | Token 成本 | 
|------|-----------|------|-----------|
| 查询改写 | 低 | 好 | 低（只改写） | 
| 历史注入 | 低 | 中（可能干扰） | 中 | 
| 两者结合 | 中 | 最好 | 高 | 
| 状态管理 | 高 | 最好 | 高 | 

## 动手实验

1. 实现一个支持多轮对话的 RAG 系统
2. 测试指代消解的效果（"它"、"那个"、"这种技术"等）
3. 对比有/无查询改写的多轮对话效果
4. 测试对话轮数增加后对回答质量的影响

## 完成标准
- [ ] 理解多轮 RAG 的核心挑战
- [ ] 实现了查询改写/指代消解
- [ ] 实现了带历史的多轮 RAG
- [ ] 评估了多轮对话中的回答质量

## 下一步 → P3-引用溯源与可信度.md
