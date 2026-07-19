# P6：中文分词与 Embedding 特化

## 目标
掌握中文 RAG 的特殊处理技巧，对比不同 Embedding 模型的中文效果

## 中文 RAG 的挑战

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| 分词粒度 | 中文不像英文有空格分隔 | 使用中文分词器 |
| 语义理解 | 中文一词多义更常见 | 选择中文优化 Embedding |
| 切分边界 | "的、了、是"等虚词影响 | 按句子/段落切 |
| 模型偏好 | 英文模型中文效果差 | 使用中文特化模型 |

## Jieba 分词

```bash
pip install jieba
```

```python
import jieba
import jieba.analyse

text = "北京大学和清华大学是中国顶尖的高等学府人工智能研究方向各具特色"

# 精确模式
words = jieba.lcut(text)
print("精确模式:", words)

# 提取关键词
keywords = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
print("关键词:")
for word, weight in keywords:
    print(f"  {word}: {weight:.4f}")

# 搜索引擎模式（更细粒度）
words_search = jieba.lcut_for_search(text)
print("搜索模式:", words_search)
```

### 自定义词典

```python
# AI 领域专业词
jieba.add_word("大语言模型")
jieba.add_word("向量数据库")
jieba.add_word("检索增强生成")
jieba.add_word("思维链")
jieba.add_word("模型微调")

text_with_terms = "大语言模型的检索增强生成技术需要结合向量数据库和思维链提示"
print(jieba.lcut(text_with_terms))
```

## 中文 Embedding 模型对比

### 在 ChromaDB 中使用 BGE 中文模型（API 方式）

SiliconFlow 同时提供 BGE-m3（多语言）和 BGE-large-zh-v1.5（中文特化），均可通过同一 API 接口调用：

```python
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI
import os

class SiliconFlowEF(EmbeddingFunction):
    """统一 Embedding Function：通过 SiliconFlow API 调用任意模型"""
    def __init__(self, model="BAAI/bge-m3", dimensions=None, query_prefix=""):
        self.model = model
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1"
        )
    
    def __call__(self, input: Documents) -> list:
        inputs = [self.query_prefix + t for t in (input if isinstance(input, list) else [input])]
        kwargs = dict(model=self.model, input=inputs)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]

# 使用 BGE-large-zh 创建集合（BGE 需加指令前缀）
bge_fn = SiliconFlowEF("BAAI/bge-large-zh-v1.5", query_prefix="为这个句子生成表示以用于检索相关文章：")
client = chromadb.PersistentClient(path="./chroma_db_bge")
collection = client.create_collection(name="bge_demo", embedding_function=bge_fn)
```

### 多模型对比实验

```python
models = {
    "BGE-m3": SiliconFlowEF("BAAI/bge-m3"),
    "BGE-large-zh": SiliconFlowEF("BAAI/bge-large-zh-v1.5", query_prefix="为这个句子生成表示以用于检索相关文章："),
    "Qwen3-Embedding": SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=1024),
}

client = chromadb.PersistentClient(path="./chroma_db_compare")
test_queries = ["人生哲理", "文学与诗歌", "励志名言"]

for name, emb_fn in models.items():
    col_name = name.replace(" ","_").replace("-","_")
    existing = [c.name for c in client.list_collections()]
    col = client.get_collection(name=col_name, embedding_function=emb_fn) if col_name in existing else None
    
    if col is None:
        col = client.create_collection(name=col_name, embedding_function=emb_fn)
        col.add(documents=texts, ids=[f"doc_{i}" for i in range(len(texts))])
    
    print(f"\n  [{name}]")
    for q in test_queries:
        r = col.query(query_texts=[q], n_results=1)
        print(f"    查询「{q}」→ {r['ids'][0][0]}: {r['documents'][0][0][:30]}...")
```

## 中文文档切分建议

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文专用切分配置
chinese_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,          # 中文 300 字符 ≈ 英文 200 token
    chunk_overlap=50,
    separators=[             # 按中文习惯优先级排列
        "\n\n",              # 段落
        "\n",                # 行
        "。", "！", "？",    # 句子结束
        "；", "，", "、",     # 短句
        " ",                 # 词
        ""                   # 字符
    ],
    length_function=len      # 使用字符数而非 token 数
)
```

## 多模型接入同一知识库

不同 Embedding 模型产生不同维度的向量。这个维度在 ChromaDB 创建集合时固定，之后不可修改。

### 模型与维度的关系

| 模型（API） | 默认维度 | 特点 |
|------|---------|------|
| BAAI/bge-m3 | 1024 | 多语言通用，支持 8192 token |
| BAAI/bge-large-zh-v1.5 | 1024 | 中文特化，512 token |
| Qwen/Qwen3-Embedding-0.6B | 4096（可调） | 支持动态维度 [64..4096] |

### 同一 chroma_db 下共存不同维度

```python
client = chromadb.PersistentClient(path="./chroma_db_excerpts")

dim_models = {
    "BGE-m3(1024d)": (SiliconFlowEF("BAAI/bge-m3"), 1024),
    "Qwen3-512d": (SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=512), 512),
    "Qwen3-256d": (SiliconFlowEF("Qwen/Qwen3-Embedding-0.6B", dimensions=256), 256),
}

test_texts = ["RAG 是检索增强生成", "向量数据库用于存储和检索高维向量"]

print("各模型向量维度:")
for name, (fn, _) in dim_models.items():
    vecs = fn(test_texts)
    print(f"  {name}: {len(vecs[0])}d  前5维: {[f'{v:.4f}' for v in vecs[0][:5]]}")

print("\n共存验证：同一 chroma_db 下不同维度集合的查询结果:")
for name, (fn, dim) in dim_models.items():
    col_name = name.replace("(","").replace(")","").replace("-","_").replace(" ","_")
    existing = [c.name for c in client.list_collections()]
    col = client.get_collection(name=col_name, embedding_function=fn) if col_name in existing else None
    if col is None:
        col = client.create_collection(name=col_name, embedding_function=fn)
        col.add(documents=texts[:10], ids=[f"doc_{i}" for i in range(len(texts[:10]))])
    r = col.query(query_texts=["人生哲理"], n_results=1)
    print(f"  {name} ({dim}d): 查询「人生哲理」→ {r['ids'][0][0]}")
```

### 切换模型的正确姿势

- 同一 ChromaDB 目录可共存不同维度的集合
- 要切换模型的集合必须删除后重建（或新建集合）
- 不同模型对同一段文本的语义理解有差异，实际检索结果也不同

## 动手实验

1. 用 Jieba 切分一篇中文技术文章，对比不用分词直接切分的区别
2. 在 ChromaDB 中配置 BGE 中文 Embedding，测试中文检索效果
3. 用同一批中文数据对比 BGE-m3 vs BGE-large-zh 的检索准确率
4. 在同一 chroma_db 下创建不同维度的集合，验证它们可以共存
5. 总结中文 RAG 的最佳实践配置

## 完成标准
- [ ] 能用 Jieba 做中文分词和关键词提取
- [ ] 能在 ChromaDB 中使用中文 Embedding 模型
- [ ] 对比了不同 Embedding 模型的中文效果
- [ ] 理解不同模型维度不同，同一 chroma_db 可共存
- [ ] 总结了中文 RAG 的配置建议

## 下一步 → [P7-本地文档问答CLI工具](P7-本地文档问答CLI工具.md)
