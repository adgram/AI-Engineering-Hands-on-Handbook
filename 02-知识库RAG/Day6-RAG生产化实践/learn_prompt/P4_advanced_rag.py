import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class SelfRAG:
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
    
    def should_retrieve(self, query: str) -> bool:
        """判断是否需要检索外部知识"""
        response = self.client.chat(
            messages=[{"role": "user", "content": f"""判断以下问题是否需要查找外部知识才能回答。

问题：{query}

如果该问题是常识性或模型训练数据中已有，输出：不需要
如果需要最新信息或特定领域知识，输出：需要
只输出"需要"或"不需要"。"""}],
        )
        decision = response.choices[0].message.content.strip()
        return "需要" in decision
    
    def is_relevant(self, query: str, doc: str) -> bool:
        """判断检索到的文档是否与问题相关"""
        response = self.client.chat(
            messages=[{"role": "user", "content": f"""判断以下文档是否与问题相关。

问题：{query}
文档：{doc[:200]}

只输出"相关"或"不相关"。"""}],
        )
        return "相关" in response.choices[0].message.content
    
    def ask(self, query: str) -> str:
        # 第一步：决定是否需要检索
        if not self.should_retrieve(query):
            response = self.client.chat(
                messages=[{"role": "user", "content": query}]
            )
            return response.choices[0].message.content
        
        # 第二步：检索
        results = self.collection.query(query_texts=[query], n_results=5)
        
        # 第三步：过滤不相关结果
        relevant_docs = []
        for doc in results['documents'][0]:
            if self.is_relevant(query, doc):
                relevant_docs.append(doc)
        
        if not relevant_docs:
            return "检索到的资料与问题不相关，无法回答。"
        
        # 第四步：用相关文档回答
        context = "\n".join(relevant_docs)
        response = self.client.chat(
            messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
        )
        return response.choices[0].message.content

# === Code Block 2 ===

class CorrectiveRAG:
    def __init__(self, collection, client, max_retries=2):
        self.collection = collection
        self.client = client
        self.max_retries = max_retries
    
    def evaluate_retrieval(self, query: str, docs: list) -> dict:
        """评估检索质量"""
        prompt = f"""评估以下检索结果是否足以回答问题。

问题：{query}

检索结果：
{"".join([f"[{i+1}] {d[:150]}\n" for i, d in enumerate(docs)])}

评估：
1. 检索结果是否与问题相关？(0-10)
2. 信息量是否足够回答问题？(0-10)
3. 是否需要重新检索？

输出 JSON：
{{"relevance": 0-10, "sufficiency": 0-10, "should_retry": true/false, "rewrite_suggestion": "改写建议"}}"""
        
        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    
    def ask(self, query: str) -> str:
        current_query = query
        
        for attempt in range(self.max_retries + 1):
            results = self.collection.query(query_texts=[current_query], n_results=3)
            docs = results['documents'][0]
            
            if not docs:
                return "未检索到相关信息。"
            
            # 评估
            eval_result = self.evaluate_retrieval(current_query, docs)
            
            if not eval_result.get("should_retry", False):
                # 质量达标，生成答案
                context = "\n".join(docs)
                resp = self.client.chat(
                    messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
                )
                return resp.choices[0].message.content
            
            # 质量不达标，重写查询
            rewrite = eval_result.get("rewrite_suggestion", "")
            if rewrite and attempt < self.max_retries:
                print(f"第{attempt+1}次检索质量不足，重写查询: {current_query} → {rewrite}")
                current_query = rewrite
        
        return "多次检索后仍无法获得足够信息。"

# === Code Block 3 ===

class AdaptiveRAG:
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
    
    def classify_question(self, query: str) -> str:
        """分类问题类型"""
        resp = self.client.chat(
            messages=[{"role": "user", "content": f"""将问题分类：
- "factual": 事实性问题（需要精确知识）
- "reasoning": 推理问题（需要逻辑分析）
- "creative": 创意问题（需要生成）
- "simple": 简单常识（不需要检索）

问题：{query}
分类："""}],
        )
        return resp.choices[0].message.content.strip()
    
    def ask(self, query: str) -> str:
        qtype = self.classify_question(query)
        print(f"[AdaptiveRAG] 问题类型: {qtype}")
        
        if qtype == "simple":
            # 简单问题，直接用 LLM
            resp = self.client.chat(
                messages=[{"role": "user", "content": query}]
            )
            return resp.choices[0].message.content
        
        elif qtype == "factual":
            # 事实问题：检索 + 严格基于资料回答
            results = self.collection.query(query_texts=[query], n_results=5)
            context = "\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
            resp = self.client.chat(
                messages=[{"role": "user", "content": f"严格基于以下资料回答问题。如果资料中没有，请说不知道。\n\n{context}\n\n问题：{query}"}],
            )
            return resp.choices[0].message.content
        
        elif qtype == "reasoning":
            # 推理问题：检索 + CoT
            results = self.collection.query(query_texts=[query], n_results=3)
            context = "\n".join(results['documents'][0])
            resp = self.client.chat(
                messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}\n\n请一步步推理。"}],
            )
            return resp.choices[0].message.content
        
        else:
            # 创意问题：宽松检索作为参考
            results = self.collection.query(query_texts=[query], n_results=2)
            context = "\n".join(results['documents'][0]) if results['documents'][0] else "无"
            resp = self.client.chat(
                messages=[{"role": "user", "content": f"参考信息：{context}\n\n问题：{query}\n\n发挥创意。"}],
            )
            return resp.choices[0].message.content

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"{Path(__file__).stem} 模块加载完成，包含 SelfRAG、CorrectiveRAG、AdaptiveRAG 类")
print(f"结果已写入 {_output_file}")
