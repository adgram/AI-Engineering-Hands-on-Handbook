import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

results = {}

# === LangChain（通过 SiliconFlow API 调用 bge-m3，无需本地模型） ===
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_classic.chains import RetrievalQA

    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1"
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=Path(__file__).parent/"chroma_db_framework")
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )
    result_lc = qa_chain.invoke({"query": "RAG 是什么？"})
    results["LangChain"] = result_lc['result'][:100]
except ImportError as e:
    results["LangChain"] = f"跳过（{e}）"

# === LlamaIndex（通过 SiliconFlow API，无需本地模型） ===
try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    Settings.embed_model = OpenAIEmbedding(
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        api_base="https://api.siliconflow.cn/v1"
    )
    Settings.llm = OpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com"
    )
    documents = [Document(text="RAG 是检索增强生成技术...")]
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    response_li = query_engine.query("RAG 是什么？")
    results["LlamaIndex"] = str(response_li)[:100]
except ImportError as e:
    results["LlamaIndex"] = f"跳过（{e}）"

# === Haystack（通过 SiliconFlow API） ===
try:
    from haystack import Pipeline, Document
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    doc_store = InMemoryDocumentStore()
    doc_store.write_documents([Document(content="RAG 是检索增强生成技术...")])
    prompt_template = """基于以下资料回答问题：
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}
问题：{{ query }}
回答："""
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(doc_store, top_k=3))
    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("llm", OpenAIGenerator(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base_url="https://api.deepseek.com",
        model="deepseek-v4-flash"
    ))
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")
    result_hs = pipeline.run({"retriever": {"query": "RAG 是什么？"}})
    results["Haystack"] = result_hs['llm']['replies'][0][:100]
except ImportError as e:
    results["Haystack"] = f"跳过（{e}）"

# 写入结果文件
_output_file = str(Path(__file__).parent / f"{Path(__file__).stem}_result.txt")
with open(_output_file, "w", encoding="utf-8") as _f:
    for k, v in results.items():
        _f.write(f"{k}: {v}\n")
print(f"结果已写入 {_output_file}")
print("\n".join([f"{k}: {v}" for k, v in results.items()]))
