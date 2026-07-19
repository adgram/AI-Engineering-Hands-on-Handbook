# P6：综合项目 — 企业级 RAG 知识库 MVP

## 目标
整合前面所学，构建一个接近生产标准的 RAG 知识库

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-P7（CLI 工具）+ Day4-P7（Pipeline）+ Day5-P8（Web 应用）+ Day6-P2（缓存）+ Day6-P3（成本监控） |
| 本章内容 | 整合 Day3-Day6 所学，构建模块化企业级 MVP（多文档格式/缓存/Token 监控/引用溯源/配置化），参考字节跳动真实业务数据设定目标。 |

## 功能要求

MVP 需要覆盖从文档接入、问答交互到监控运维的完整链路，以下列出核心功能清单。

- [ ] 多文档格式支持（TXT / MD / PDF / DOCX）
- [ ] 后台管理界面
- [ ] 全文问答 + 引用溯源
- [ ] 多轮对话
- [ ] Token 用量统计与成本监控
- [ ] 缓存（精确 + 语义）
- [ ] 检索质量评估仪表盘

## 项目结构

项目采用前后端分离的模块化设计，以下为完整的目录树及各模块的职责说明。

```
enterprise_rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置
│   ├── rag/
│   │   ├── engine.py        # RAG 主引擎
│   │   ├── retriever.py     # 检索器（含高级模式）
│   │   ├── generator.py     # 生成器
│   │   ├── reranker.py      # 重排序
│   │   └── cache.py         # 缓存
│   ├── document/
│   │   ├── loader.py        # 文档加载
│   │   ├── chunker.py       # 文档切分
│   │   └── indexer.py       # 索引管理
│   ├── monitor/
│   │   ├── token_tracker.py # Token 监控
│   │   └── evaluator.py     # 质量评估
│   └── api/
│       ├── routes.py        # API 路由
│       └── models.py        # 数据模型
├── frontend/
│   └── app.py               # Streamlit 管理界面
├── data/                    # 数据存储
├── requirements.txt
└── .env
```

## 核心代码框架

MVP 采用模块化分层架构，核心模块包括 RAG 引擎、文档管理、监控统计和 API 路由四个部分，各模块职责清晰、可独立迭代。

### app/rag/engine.py

> `query` 方法的检索+Prompt+LLM 逻辑与 Day5-P8 `RAGEngine.ask` 基本相同（详见 Day5-P8），此处仅展示新增的缓存和统计功能；`add_document` 的切分+入库逻辑同 Day3-P7。

```python
import os, json, time
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
load_dotenv()

class EnterpriseRAG:
    def __init__(self, persist_dir="./data/chroma"):
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection("knowledge")
        self.cache = {}
        self.stats = {"total_calls": 0, "total_tokens": 0}
    
    def query(self, question: str, k: int = 5, use_cache: bool = True) -> dict:
        """问答（检索+Prompt+LLM 同 Day5-P8），新增缓存和统计"""
        if use_cache and question in self.cache:
            self.stats["total_calls"] += 1
            return self.cache[question]
        results = self.collection.query(query_texts=[question], n_results=k)
        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}
        context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": "你是企业知识库助手,基于资料回答问题并标注来源。"},
                      {"role": "user", "content": f"参考资料：{context}\n\n问题：{question}"}],
        )
        answer = response.choices[0].message.content
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += response.usage.total_tokens
        result = {"answer": answer, "sources": [d[:200] for d in results['documents'][0]], "tokens": response.usage.total_tokens}
        if use_cache:
            self.cache[question] = result
        return result
    
    def add_document(self, content: str, metadata: dict = None) -> int:
        from app.document.chunker import chunk_document
        chunks = chunk_document(content)
        ids = [f"doc_{int(time.time())}_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids, metadatas=[metadata or {}] * len(chunks))
        return len(chunks)
    
    def get_stats(self):
        return {**self.stats, "cache_size": len(self.cache), "doc_count": self.collection.count()}
```

### app/api/routes.py

API 路由层基于 FastAPIRouter 实现，提供问答、文档添加、文件上传和统计查询四个端点。

```python
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.rag.engine import EnterpriseRAG

router = APIRouter()
rag = EnterpriseRAG()

class AskRequest(BaseModel):
    question: str
    k: int = 5

class DocumentInput(BaseModel):
    content: str
    metadata: dict = {}

@router.post("/ask")
async def ask(req: AskRequest):
    return rag.query(req.question, req.k)

@router.post("/documents")
async def add_document(req: DocumentInput):
    chunks = rag.add_document(req.content, req.metadata)
    return {"status": "ok", "chunks": chunks}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="ignore")
    chunks = rag.add_document(content, {"filename": file.filename})
    return {"filename": file.filename, "chunks": chunks}

@router.get("/stats")
async def stats():
    return rag.get_stats()
```

## 验收清单

- [ ] 完整的项目结构（模块化/可扩展）
- [ ] API 接口（文档上传/问答/统计）
- [ ] 管理界面（文档管理/问答/统计展示）
- [ ] 缓存机制
- [ ] Token 监控
- [ ] 引用溯源
- [ ] 配置化（.env / config.py）

## 工业参考：字节跳动 RAG 业务落地效果

构建 MVP 时可以参考这些真实业务数据来设定目标：

| 业务线 | 数据规模 | 核心指标 | 优化效果 |
|-------|---------|---------|----------|
| 抖音电商客服 | 5000万SKU，日均500万咨询 | 响应300ms，准确率95% | 客服效率↑10x，成本↓50% |
| 飞书知识库 | 5亿文档，50亿向量 | 召回率92%，满意度91% | 找文档时间从15min→2min |
| 金融研报 | 日均1万份研报，100亿向量 | 准确率96%，延迟<1.5s | 分析师效率↑6x |

```python
class MVPBenchmark:
    """用字节跳动案例数据作为MVP的参考基准"""
    
    @staticmethod
    def estimate_target(scale: str, scene: str) -> dict:
        """根据规模和场景估算应该达到的目标"""
        benchmarks = {
            "small": {
                "docs": 1000,
                "latency_ms": 500,
                "recall": 0.85,
                "accuracy": 0.90,
            },
            "medium": {
                "docs": 100000,
                "latency_ms": 800,
                "recall": 0.90,
                "accuracy": 0.92,
            },
            "large": {
                "docs": 10000000,
                "latency_ms": 1500,
                "recall": 0.92,
                "accuracy": 0.95,
            },
        }
        return benchmarks.get(scale, benchmarks["small"])

# 你的MVP至少达到 small 级别指标
target = MVPBenchmark.estimate_target("small", "qa")
print(f"MVP目标：延迟<{target['latency_ms']}ms, 召回率>{target['recall']}, 准确率>{target['accuracy']}")
```

## 下一步 → Agent核心概念
