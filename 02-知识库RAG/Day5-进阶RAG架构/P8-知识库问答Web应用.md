# P8：小项目 — 知识库问答 Web 应用

## 目标
将 RAG Pipeline 封装为 Web 应用，支持文档上传和交互式问答

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day3-P7（CLI 工具）+ Day4-P7（带评估的 Pipeline） |
| 本章内容 | 用 FastAPI + Streamlit 把 RAG 封装为 Web 应用，支持文档上传、交互问答、来源展示和参数调节，从开发者工具升级为用户产品。 |

## 技术栈

本项目采用以下技术栈构建，覆盖后端 API、前端界面、向量存储和 LLM 调用四个层面。

- 后端：FastAPI
- 前端：Streamlit
- 向量库：ChromaDB
- LLM：DeepSeek API

## 项目结构

项目按前后端分离组织，backend 目录存放 FastAPI 服务代码，frontend 目录存放 Streamlit 前端界面，根目录存放依赖和环境配置。

```
rag_web/
├── backend/
│   ├── main.py          # FastAPI 服务
│   ├── rag_engine.py    # RAG 引擎
│   ├── document.py      # 文档处理
│   └── models.py        # 数据模型
├── frontend/
│   └── app.py           # Streamlit 前端
├── requirements.txt
└── .env
```

### backend/rag_engine.py

> `add_document`（文档切分 + ChromaDB 添加）详见 Day3-P7，`ask`（检索 + Prompt 构建 + LLM 生成）详见 Day4-P1/P7，此处不再重复，仅展示 Web 场景特有的 API 封装：

```python
import os, json
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
load_dotenv()

class RAGEngine:
    def __init__(self, persist_dir="./rag_data"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection("knowledge_base")
    
    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """添加文档到知识库（切分逻辑同 Day3-P7）"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "。", "，", " ", ""])
        chunks = splitter.split_text(content)
        ids = [f"{doc_id}#{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk": i, **(metadata or {})} for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)
    
    def ask(self, question: str, k: int = 5) -> dict:
        """问答（Prompt 模板同 Day4-P1）"""
        results = self.collection.query(query_texts=[question], n_results=k)
        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}
        context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "system", "content": "你是知识库助手，基于资料回答问题。标注引用来源。"},
                      {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}],
        )
        return {"answer": response.choices[0].message.content,
                "sources": [{"content": d[:200], "doc_id": m.get("doc_id", "unknown")}
                           for d, m in zip(results['documents'][0], results['metadatas'][0])]}
    
    def list_documents(self) -> list:
        meta = self.collection.get()
        return list(set(m.get("doc_id", "unknown") for m in meta['metadatas']))
```

### backend/main.py

后端 API 主文件，定义文档上传、问答和文档列表三个接口，通过 CORS 中间件支持跨域请求。

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI(title="RAG Knowledge Base API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = RAGEngine()

class AskRequest(BaseModel):
    question: str
    k: int = 5

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    chunks = engine.add_document(file.filename, content)
    return {"filename": file.filename, "chunks": chunks}

@app.post("/ask")
async def ask(req: AskRequest):
    return engine.ask(req.question, req.k)

@app.get("/documents")
async def list_docs():
    return {"documents": engine.list_documents()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### frontend/app.py

Streamlit 前端界面，提供文档上传管理、问答输入和来源展示功能，通过 HTTP 请求调用后端 API。

```python
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG 知识库问答", layout="wide")
st.title("📚 RAG 知识库问答系统")

# 侧边栏：文档管理
with st.sidebar:
    st.header("文档管理")
    uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "pdf"])
    if uploaded_file and st.button("添加到知识库"):
        files = {"file": (uploaded_file.name, uploaded_file.read(), "text/plain")}
        resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.ok:
            st.success(f"已添加 {resp.json()['chunks']} 个文档块")
    
    st.divider()
    st.header("已索引文档")
    resp = requests.get(f"{API_URL}/documents")
    if resp.ok:
        for doc in resp.json()["documents"]:
            st.text(f"📄 {doc}")

# 主区域：问答
st.header("问答")
question = st.text_input("输入你的问题")

col1, col2 = st.columns([1, 4])
with col1:
    k = st.number_input("检索数量", min_value=1, max_value=20, value=5)

if question:
    with st.spinner("思考中..."):
        resp = requests.post(f"{API_URL}/ask", json={"question": question, "k": k})
    
    if resp.ok:
        result = resp.json()
        st.markdown(result["answer"])
        
        with st.expander(f"查看来源 ({len(result['sources'])} 项)"):
            for src in result["sources"]:
                st.text(f"📄 {src['doc_id']}")
                st.caption(src["content"][:200])
```

### requirements.txt

项目所需 Python 依赖包列表，包括 Web 框架、向量库、LLM SDK 和文档处理工具。

```
fastapi
uvicorn
streamlit
chromadb
openai
python-dotenv
langchain-text-splitters
requests
python-multipart
```

## 运行

分别启动后端和前端服务，后端运行在 8000 端口提供 API，前端通过 Streamlit 运行在默认端口提供交互界面。

```bash
# 终端1: 启动后端
cd backend
pip install -r ../requirements.txt
python main.py

# 终端2: 启动前端
cd frontend
streamlit run app.py
```

## 验收清单

- [ ] 后端 API 可用（文档上传 / 问答 / 文档列表）
- [ ] 前端可上传文档并添加到知识库
- [ ] 前端可输入问题并看到回答
- [ ] 回答带来源展示
- [ ] 可调节检索参数

## 下一步 → [P9-A-RAG复现实战](P9-A-RAG复现实战.md)
