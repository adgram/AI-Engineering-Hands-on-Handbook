from typing import Optional
from chromadb import Documents, EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SiliconFlowEmbedding(EmbeddingFunction):
    def __init__(self, model: str = "BAAI/bge-m3"):
        self.client: OpenAI
        self.model = model

    def __call__(self, input: Documents) -> list[list[float]]:
        pass


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "knowledge",
                 embedding_fn: Optional[EmbeddingFunction] = None):
        self.client = None
        self.collection = None

    def add(self, texts: list[str], ids: list[str],
            metadatas: Optional[list[dict]] = None, batch_size: int = 100):
        pass

    def search(self, query: str, n_results: int = 5,
               where_filter: Optional[dict] = None):
        pass

    def count(self) -> int:
        pass


class Chunker:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter: RecursiveCharacterTextSplitter = None

    def chunk_text(self, file_path: str, text: str) -> list[dict]:
        pass

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        pass


def load_text(file_path: str) -> str:
    pass


def load_pdf(file_path: str) -> str:
    pass


def load_document(file_path: str) -> str:
    pass


def load_directory(dir_path: str) -> list[dict]:
    pass


class BaseRAG:
    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "knowledge",
        embedding_fn: Optional[EmbeddingFunction] = None,
        llm_model: str = "deepseek-v4-flash",
    ):
        self.llm: OpenAI  = None
        self.llm_model = llm_model
        self.store: VectorStore  = None
        self.cache = {}
        self.stats = {"total_calls": 0, "total_tokens": 0}

    def add_documents(self, documents: list[str],
                      metadatas: Optional[list[dict]] = None,
                      ids: Optional[list[str]] = None):
        pass

    def add_chunks(self, chunks: list[dict]):
        pass

    def query(self, question: str, k: int = 5, use_cache: bool = True) -> dict:
        pass

    def get_stats(self) -> dict:
        pass

    def clear_cache(self):
        pass
