from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

_DIR = Path(__file__).parent
CHROMA_PATH = str(_DIR / "chroma_db_doc_qa")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-flash"
TOP_K = 5
