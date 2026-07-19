import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.rag_client import BaseRAG, load_directory
from dotenv import load_dotenv
load_dotenv()

# 初始化 RAG（从 rag_knowledge 目录加载文档）
_BASE = Path(__file__).parent.parent.parent.parent
data_dir = str(_BASE / "common" / "text_data" / "rag_knowledge")
db_path = str(Path(__file__).parent / "chroma_db_p6")

rag = BaseRAG(persist_dir=db_path, collection_name="incremental_update_demo")
if rag.store.count() == 0:
    loaded = load_directory(data_dir)
    rag.add_documents(
        documents=[d["content"] for d in loaded],
        metadatas=[{"source": "rag_knowledge", "topic": d.get("topic", "general")} for d in loaded],
        ids=[f"doc_{i+1}" for i in range(len(loaded))]
    )
    print(f"知识库初始化完成，已添加 {rag.store.count()} 条文档")

def rebuild_index(collection, all_docs: list):
    """删除旧索引，重新添加所有文档"""
    # 获取所有现有 ID
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids=existing['ids'])
    
    # 重新添加
    collection.add(
        documents=[d['content'] for d in all_docs],
        metadatas=[d.get('metadata', {}) for d in all_docs],
        ids=[d['id'] for d in all_docs]
    )
    print(f"索引重建完成，共 {len(all_docs)} 条")

# === Code Block 2 ===

def incremental_add(collection, new_docs: list, check_duplicate: bool = True):
    """增量添加新文档"""
    existing_ids = set(collection.get()['ids']) if check_duplicate else set()
    
    added = 0
    for doc in new_docs:
        if doc['id'] not in existing_ids:
            collection.add(
                documents=[doc['content']],
                metadatas=[doc.get('metadata', {})],
                ids=[doc['id']]
            )
            added += 1
        else:
            print(f"跳过已存在的文档: {doc['id']}")
    
    print(f"新增 {added} 条，当前总数 {collection.count()}")
    return added

# === Code Block 3 ===

def update_document(collection, doc_id: str, new_content: str, new_metadata: dict = None):
    """更新文档内容"""
    try:
        # ChromaDB 的 update 方法
        collection.update(
            ids=[doc_id],
            documents=[new_content],
            metadatas=[new_metadata] if new_metadata else None
        )
        print(f"文档 {doc_id} 已更新")
    except Exception as e:
        print(f"更新失败: {e}")
        # 如果 update 失败，尝试 upsert
        collection.upsert(
            ids=[doc_id],
            documents=[new_content],
            metadatas=[new_metadata]
        )

# === Code Block 4 ===

import time
import os
from datetime import datetime
import hashlib

class FileSyncIndexer:
    """监控文件变化，自动更新索引"""
    
    def __init__(self, collection, watch_dir: str):
        self.collection = collection
        self.watch_dir = watch_dir
        self.file_hashes = {}  # 文件路径 → MP5
    
    def _file_hash(self, path: str) -> str:
        return hashlib.md5(open(path, "rb").read()).hexdigest()
    
    def sync(self):
        """同步文件系统到向量索引"""
        current_files = set()
        
        for root, _, files in os.walk(self.watch_dir):
            for f in files:
                if f.endswith(('.txt', '.md', '.pdf')):
                    path = os.path.join(root, f)
                    current_files.add(path)
                    new_hash = self._file_hash(path)
                    
                    if path not in self.file_hashes:
                        print(f"[新增] {path}")
                        self._add_file(path)
                    elif self.file_hashes[path] != new_hash:
                        print(f"[修改] {path}")
                        doc_id = f"file_{path.replace(os.sep, '_')}"
                        with open(path, "r", encoding="utf-8") as fh:
                            content = fh.read()
                        update_document(self.collection, doc_id, content)
                    
                    self.file_hashes[path] = new_hash
        
        # 检测删除
        for path in list(self.file_hashes.keys()):
            if path not in current_files:
                print(f"[删除] {path}")
                doc_id = f"file_{path.replace(os.sep, '_')}"
                try:
                    self.collection.delete(ids=[doc_id])
                except:
                    pass
                del self.file_hashes[path]
    
    def _add_file(self, path: str):
        doc_id = f"file_{path.replace(os.sep, '_')}"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.collection.add(
            documents=[content],
            metadatas=[{"file": path, "updated": datetime.now().isoformat()}],
            ids=[doc_id]
        )
    
    def watch(self, interval: int = 60):
        """定时监控文件变化"""
        print(f"开始监控目录: {self.watch_dir}")
        while True:
            self.sync()
            time.sleep(interval)

# 使用
collection = rag.store
indexer = FileSyncIndexer(collection, Path(__file__).parent/"watch_docs")
indexer.sync()  # 一次性同步
# indexer.watch(interval=60)  # 持续监控

# === Code Block 5 ===

class VersionedVectorStore:
    """带版本管理的向量库"""
    
    def __init__(self, base_collection):
        self.collection = base_collection
        self.version_meta = {"version": 1, "last_updated": None}
    
    def add_version(self, docs: list):
        """添加新版本"""
        self.version_meta["version"] += 1
        self.version_meta["last_updated"] = datetime.now().isoformat()
        
        for doc in docs:
            doc['metadata']['version'] = self.version_meta["version"]
            self.collection.upsert(
                ids=[doc['id']],
                documents=[doc['content']],
                metadatas=[doc['metadata']]
            )
    
    def rollback(self, target_version: int):
        """回滚到指定版本（删除该版本之后添加的文档）"""
        # 获取所有文档中版本大于 target_version 的
        all_docs = self.collection.get()
        to_delete = []
        for i, meta in enumerate(all_docs['metadatas']):
            if meta.get('version', 1) > target_version:
                to_delete.append(all_docs['ids'][i])
        
        if to_delete:
            self.collection.delete(ids=to_delete)
        
        self.version_meta["version"] = target_version
        print(f"回滚到 v{target_version}，删除了 {len(to_delete)} 条")


# 写入结果文件
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(_output_file, "w", encoding="utf-8") as _f:
    _f.write(f"文件同步索引器已运行，监控目录: {indexer.watch_dir}")
print(f"结果已写入 {_output_file}")