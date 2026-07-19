import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import json
from collections import defaultdict

class SimpleGraphRAG:
    """简单版 GraphRAG：实体提取 + 关系构建 + 图检索"""
    
    def __init__(self, collection, client):
        self.collection = collection
        self.client = client
        self.graph = defaultdict(list)  # entity → [related_entities]
        self.entity_docs = defaultdict(list)  # entity → [doc_ids]
    
    def extract_entities(self, text: str) -> list:
        """从文本中提取实体"""
        resp = self.client.chat(
            messages=[{"role": "user", "content": f"""从以下文本中提取所有实体（人名、技术名词、概念等），以JSON数组格式输出。

文本：{text[:500]}

输出：["实体1", "实体2", ...]"""}],
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            # JSON Mode 返回对象，兼容 {"entities": [...]} 或裸数组两种格式
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("entities", "entity", "items", "result"):
                    val = data.get(key)
                    if isinstance(val, list):
                        return val
            return []
        except:
            return []
    
    def build_graph(self, texts: list):
        """从文档构建知识图谱"""
        for doc_id, text in enumerate(texts):
            entities = self.extract_entities(text)
            for e in entities:
                self.entity_docs[e].append(doc_id)
            
            # 同一文档中的实体建立连接
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    self.graph[entities[i]].append(entities[j])
                    self.graph[entities[j]].append(entities[i])
        
        print(f"图谱构建完成: {len(self.graph)} 个实体")
    
    def expand_query(self, query: str) -> str:
        """通过知识图谱扩展查询"""
        # 提取查询中的实体
        entities = self.extract_entities(query)
        
        # 在图中查找相关实体
        related = set()
        for e in entities:
            for rel in self.graph.get(e, []):
                related.add(rel)
        
        if related:
            context = f"相关问题涉及：{', '.join(entities)}\n"
            context += f"相关概念：{', '.join(list(related)[:5])}"
            return f"{query}\n\n{context}"
        
        return query
    
    def ask(self, query: str) -> str:
        # 用图谱扩展查询
        expanded_query = self.expand_query(query)
        
        # 检索
        results = self.collection.query(query_texts=[expanded_query], n_results=5)
        
        if not results['documents'][0]:
            return "未找到相关信息。"
        
        context = "\n".join(results['documents'][0])
        
        # 生成
        resp = self.client.chat(
            messages=[{"role": "user", "content": f"资料：{context}\n\n问题：{query}"}]
        )
        return resp.choices[0].message.content

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write("P5_graphrag 模块加载完成，包含 SimpleGraphRAG 类（实体提取 + 图检索）")
print(f"结果已写入 {_output_file}")
