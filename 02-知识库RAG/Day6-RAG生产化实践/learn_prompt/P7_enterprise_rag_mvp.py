import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.llm_client import LLMClient
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EnterpriseRAG:
    def __init__(self, persist_dir="chroma_db_enterprise"):
        self.client = LLMClient()
        self.chroma = chromadb.PersistentClient(path=Path(__file__).parent/persist_dir)
        self.collection = self.chroma.get_or_create_collection("knowledge")
        self.cache = {}  # 简单缓存
        self.stats = {"total_calls": 0, "total_tokens": 0}
    
    def query(self, question: str, k: int = 5, use_cache: bool = True) -> dict:
        if use_cache and question in self.cache:
            self.stats["total_calls"] += 1
            return self.cache[question]
        
        results = self.collection.query(query_texts=[question], n_results=k)
        if not results['documents'][0]:
            return {"answer": "未找到相关信息", "sources": []}
        
        context = "\n\n".join([f"[{i+1}] {d}" for i, d in enumerate(results['documents'][0])])
        response = self.client.chat(
            messages=[
                {"role": "system", "content": "你是企业知识库助手,基于资料回答问题并标注来源。"},
                {"role": "user", "content": f"参考资料：{context}\n\n问题：{question}"}
            ],
        )
        
        answer = response.choices[0].message.content
        self.stats["total_calls"] += 1
        self.stats["total_tokens"] += response.usage.total_tokens
        
        result = {
            "answer": answer,
            "sources": [d[:200] for d in results['documents'][0]],
            "tokens": response.usage.total_tokens
        }
        
        if use_cache:
            self.cache[question] = result
        return result
    
    def add_document(self, content: str, metadata: dict = None) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_text(content)
        ids = [f"doc_{int(time.time())}_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[dict(metadata or {}) for _ in chunks]
        )
        return len(chunks)
    
    def get_stats(self):
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "doc_count": self.collection.count()
        }

# === Code Block 2 ===

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

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

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write("P7_enterprise_rag_mvp 模块加载完成，包含 EnterpriseRAG 类及 FastAPI 路由")
print(f"结果已写入 {_output_file}")
