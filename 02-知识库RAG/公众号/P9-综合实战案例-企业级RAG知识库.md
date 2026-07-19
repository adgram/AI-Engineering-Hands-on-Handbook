# 知识库RAG | P9：综合实战案例 — 企业级 RAG 知识库

## 前言

本节将结合 P1-P8 核心技术（并引入 P10 拓展的缓存与成本优化），实现一个完整可用的企业级 RAG 知识库——支持多文档格式、Web 界面、引用溯源和质量监控。这是 RAG 部分唯一的综合实战案例，把前面学到的 Embedding、向量数据库、文档切分、混合检索、Rerank、Prompt 注入、质量评估、多轮对话、查询优化、增量更新等技术整合为一个可运行的产品。

注：本案例由 AI 实现，需要自定义可以让 AI 进行修改。

## 项目说明

构建一个前后端分离的知识库问答系统，用户上传文档后即可通过自然语言提问，系统检索相关内容并生成带来源标注的回答。

### 核心流程

```
用户提问（自然语言）
    ↓
① 文档加载（P3 切分）→ 支持 TXT/MD/PDF/DOCX，递归字符切分
    ↓
② 向量化存储（P1+P2）→ BAAI/bge-m3 Embedding + ChromaDB 持久化
    ↓
③ 检索（P4+P7）→ 元数据过滤 + 多路召回 + RRF 融合 + Rerank 精排
    ↓
④ 上下文注入（P5）→ 按相关性排列 + 五段式 Prompt 模板
    ↓
⑤ 缓存检查（拓展）→ 精确缓存 → 语义缓存 → 调用 LLM
    ↓
⑥ 生成（P5）→ DeepSeek 生成带引用标注的回答
    ↓
⑦ 质量监控（P6）→ Token 统计 + 质量评估
    ↓
⑧ 返回结果 → Web 界面展示回答 + 来源列表
```

## 功能清单

- 多文档格式支持（TXT/MD/PDF/DOCX）
- 后台文档管理（上传、列表、删除）
- 全文问答 + 引用溯源
- 多轮对话支持
- Token 用量统计
- 缓存机制（精确 + 语义）
- 质量评估仪表盘

## 项目结构

前后端分离架构：

```
enterprise_rag/
├── app/
│   ├── rag/
│   │   ├── engine.py         # RAG 主引擎（整合检索+生成+缓存+统计）
│   │   ├── retriever.py      # 多策略检索（向量/HyDE/多路）
│   │   ├── generator.py      # 生成器（standard/citation 两种 Prompt）
│   │   ├── reranker.py       # API 重排序
│   │   └── cache.py          # 缓存（精确+语义）
│   ├── document/
│   │   ├── loader.py         # 文档加载器（多格式）
│   │   ├── chunker.py        # 文档切分器
│   │   └── indexer.py        # 索引管理（增量更新）
│   ├── monitor/
│   │   ├── token_tracker.py  # Token 用量监控
│   │   └── evaluator.py      # 质量评估
│   └── api/
│       ├── routes.py         # FastAPI 路由
│       └── models.py         # 请求/响应模型
├── frontend/
│   └── app.py                # Streamlit 前端
├── config.py                 # 集中配置
└── .env
```

## 核心代码

### config.py — 集中配置

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Embedding 配置
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"

# LLM 配置
LLM_MODEL = "deepseek-v4-flash"
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"

# RAG 配置
CHROMA_PATH = "chroma_db_enterprise"
COLLECTION_NAME = "enterprise_kb"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_TOP_N = 3
SIMILARITY_THRESHOLD = 0.7

# 缓存配置
CACHE_DIR = "./cache"
SEMANTIC_CACHE_THRESHOLD = 0.3  # 语义缓存相似度阈值
```

### engine.py — RAG 主引擎

```python
class EnterpriseRAG:
    """企业级 RAG 引擎：整合检索+生成+缓存+统计"""

    def __init__(self):
        # 初始化各模块（依赖注入）
        self.store = VectorStore(CHROMA_PATH, COLLECTION_NAME, SiliconFlowEF())
        self.retriever = MultiRouteRetriever(self.store, self.llm)
        self.reranker = ApiReranker(EMBEDDING_API_KEY)
        self.cache = RAGCache(CACHE_DIR, self.store)
        self.tracker = TokenMonitor()
        self.generator = Generator(self.llm)

    def query(self, question: str, k: int = TOP_K) -> dict:
        # 1. 检查缓存
        cached = self.cache.get(question)
        if cached:
            self.tracker.record_cache_hit()
            return cached

        # 2. 多路检索 + RRF 融合
        retrieved = self.retriever.multi_route_search(question, k=k * 2)

        # 3. Rerank 精排
        reranked = self.reranker.rerank(question, retrieved['documents'], top_n=RERANK_TOP_N)

        # 4. 动态筛选（相似度 < 0.7 的丢弃）
        filtered = [d for d in reranked if d['score'] >= SIMILARITY_THRESHOLD]

        # 5. 五段式 Prompt 生成
        messages = self.generator.build_citation_prompt(question, filtered)

        # 6. LLM 生成
        response = self.llm.chat.completions.create(model=LLM_MODEL, messages=messages)
        answer = response.choices[0].message.content

        # 7. 统计 + 缓存
        self.tracker.record(response.usage)
        result = {"answer": answer, "sources": filtered, "tokens": response.usage.total_tokens}
        self.cache.set(question, result)
        return result

    def add_document(self, file_path: str) -> int:
        """加载+切分+入库，返回文档块数"""
        content = DocumentLoader.load(file_path)
        chunks = Chunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(content)
        self.store.add_chunks(chunks, metadata={"file": file_path})
        return len(chunks)

    def get_stats(self) -> dict:
        return self.tracker.report()
```

### generator.py — 带引用的 Prompt 生成

```python
class Generator:
    def build_citation_prompt(self, query: str, contexts: list) -> list:
        """五段式提示模板 + 引用标注"""
        context_str = "\n".join([
            f"[{i+1}] {c['content'][:200]}（相似度：{c['score']:.2f}）"
            for i, c in enumerate(contexts)
        ])
        return [
            {"role": "system", "content": f"""【角色定义】你是专业的知识库问答助手，需基于提供的检索信息回答问题。
【任务指令】用户的问题是：{query}，请结合以下检索信息，生成准确、简洁的回答。
【检索信息】
{context_str}
【格式要求】回答需分点说明，每点不超过30字。引用资料时标注来源编号如[1]。若信息不足，请说明"当前信息不足以完全回答"。
【示例引导】
用户查询：抖音小店入驻需多少钱？
检索信息：1. 入驻需缴纳2000元保证金
参考回答：入驻需缴2000元保证金[1]"""},
            {"role": "user", "content": query}
        ]
```

### cache.py — 精确 + 语义缓存

```python
class RAGCache:
    """L1 精确缓存 → L2 语义缓存 → L3 调用 API"""
    def __init__(self, cache_dir, store):
        self.exact_cache = diskcache.Cache(f"{cache_dir}/exact")
        self.store = store  # 用 ChromaDB 做语义缓存

    def get(self, query: str) -> dict:
        # L1: 精确匹配
        key = hashlib.sha256(query.encode()).hexdigest()
        if key in self.exact_cache:
            return self.exact_cache[key]

        # L2: 语义匹配（相似度低于阈值即命中）
        results = self.store.collection.query(query_texts=[query], n_results=1)
        if results['distances'][0] and results['distances'][0][0] < SEMANTIC_CACHE_THRESHOLD:
            return json.loads(results['metadatas'][0][0]['cached_result'])
        return None

    def set(self, query: str, result: dict):
        key = hashlib.sha256(query.encode()).hexdigest()
        self.exact_cache[key] = result
        # 同时写入语义缓存库
        self.store.collection.add(
            documents=[query],
            metadatas=[{"cached_result": json.dumps(result, ensure_ascii=False)}],
            ids=[f"cache_{key}"]
        )
```

### api/routes.py — FastAPI 后端

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="企业级 RAG 知识库")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

rag = EnterpriseRAG()

@app.post("/ask")
async def ask(question: str):
    result = rag.query(question)
    return {"answer": result["answer"], "sources": result["sources"]}

@app.post("/documents")
async def add_document(file: UploadFile = File(...)):
    path = f"./uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    count = rag.add_document(path)
    return {"file": file.filename, "chunks": count}

@app.get("/stats")
async def stats():
    return rag.get_stats()
```

### frontend/app.py — Streamlit 前端

```python
import streamlit as st
import requests

st.title("📚 企业级 RAG 知识库")

# 侧边栏：文档管理
with st.sidebar:
    st.header("📁 文档管理")
    uploaded = st.file_uploader("上传文档", type=["txt", "md", "pdf", "docx"])
    if uploaded and st.button("入库"):
        requests.post("http://localhost:8000/documents", files={"file": uploaded})
        st.success("文档已入库")

# 主区域：问答
question = st.text_input("提问：")
if st.button("提问") and question:
    with st.spinner("检索中..."):
        res = requests.post(f"http://localhost:8000/ask?question={question}").json()
    st.write(res["answer"])
    if res["sources"]:
        st.subheader("来源")
        for i, src in enumerate(res["sources"]):
            st.write(f"[{i+1}] {src['content'][:100]}... (相似度: {src['score']:.2f})")
```

## 运行方式

```bash
# 安装依赖
pip install fastapi uvicorn streamlit chromadb openai python-dotenv \
    langchain-text-splitters pymupdf openpyxl diskcache numpy requests

# 配置 .env（DEEPSEEK_API_KEY + SILICONFLOW_API_KEY）

# 启动后端（端口 8000）
uvicorn app.api.routes:app --reload --port 8000

# 启动前端（另一终端）
streamlit run frontend/app.py
```

浏览器打开 Streamlit 提示的地址（通常 `http://localhost:8501`）即可使用。
