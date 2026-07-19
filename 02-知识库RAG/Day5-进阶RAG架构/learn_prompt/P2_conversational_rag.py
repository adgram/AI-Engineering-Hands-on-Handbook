import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory, resolve_query_with_history
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p2")

rag = BaseRAG(persist_dir=db_path, collection_name="conversational_rag_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

# 使用 common.rag_client 中的 resolve_query_with_history 函数
# 这里直接调用，无需重复定义

# 测试
history = [
    {"user": "什么是 RAG？", "assistant": "RAG 是检索增强生成..."},
    {"user": "它有什么优点？", "assistant": "优点包括减少幻觉..."},
]

rewritten = resolve_query_with_history(rag.llm, "它的工作原理是什么？", history)
print(f"原问题: 它的工作原理是什么？")
print(f"改写后: {rewritten}")

# === Code Block 2 ===

class ConversationalRAG:
    def __init__(self, rag, max_history=3):
        self.rag = rag
        self.collection = rag.store
        self.max_history = max_history
        self.history = []
    
    def ask(self, query: str) -> str:
        # 改写查询（如果需要）
        if self.history:
            standalone_query = resolve_query_with_history(self.rag.llm, query, self.history)
        else:
            standalone_query = query
        
        # 检索
        results = self.collection.search(standalone_query, n_results=3)
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
        
        response = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=messages,
        )
        
        answer = response.choices[0].message.content
        
        # 保存历史
        self.history.append({"user": query, "assistant": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return answer
    
    def _format_history(self) -> str:
        lines = []
        for h in self.history[-self.max_history:]:
            lines.append(f"用户: {h['user']}")
            lines.append(f"助手: {h['assistant']}")
        return "\n".join(lines)

# 测试多轮对话
conv_rag = ConversationalRAG(rag)

print(conv_rag.ask("RAG 是什么？"))
print("\n--- 第二轮 ---")
print(conv_rag.ask("它有什么优势？"))
print("\n--- 第三轮 ---")
last_response = conv_rag.ask("和微调相比呢？"); print(last_response)

# === Code Block 3 ===

class ConversationState:
    """管理对话状态和临时信息"""
    
    def __init__(self):
        self.current_topic = None
        self.mentioned_entities = []
        self.pending_clarification = None
    
    def update(self, query: str, answer: str, retrieved_docs: list):
        """更新对话状态"""
        # 提取当前主题
        resp = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[{"role": "user", "content": f"从以下对话中提取当前主题词（1-3个词）：\n用户：{query}\nAI：{answer}"}],
        )
        topic = resp.choices[0].message.content.strip()
        if topic and topic != "无":
            self.current_topic = topic


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"重写后查询: {rewritten}\n最终回答: {last_response}")
print(f"结果已写入 {_output_file}")