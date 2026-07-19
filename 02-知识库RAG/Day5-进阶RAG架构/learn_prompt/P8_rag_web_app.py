import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p8")

rag = BaseRAG(persist_dir=db_path, collection_name="rag_web_app_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

class RAGEngine:
    def __init__(self, rag):
        self.rag = rag
        self.collection = rag.store.collection  # 直接使用 ChromaDB 原生接口
    
    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """添加文档到知识库"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_text(content)
        
        ids = [f"{doc_id}#{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk": i, **(metadata or {})} for i in range(len(chunks))]
        
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)
    
    def ask(self, question: str, k: int = 5) -> dict:
        results = self.collection.query(query_texts=[question], n_results=k)
        
        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}
        
        context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
        
        response = self.rag.llm.chat.completions.create(
            model=self.rag.llm_model,
            messages=[
                {"role": "system", "content": "你是知识库助手，基于资料回答问题。标注引用来源。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
            ],
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {"content": d[:200], "doc_id": m.get("doc_id", "unknown")}
                for d, m in zip(results['documents'][0], results['metadatas'][0])
            ]
        }
    
    def list_documents(self) -> list:
        meta = self.collection.get()
        doc_ids = set(m.get("doc_id", "unknown") for m in meta['metadatas'])
        return list(doc_ids)

# === Code Block 2 ===

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Base API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = RAGEngine(rag)

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

# === Code Block 3 ===

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

# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"RAG Web应用已配置，路由数量: {len(app.routes)}")
print(f"结果已写入 {_output_file}")