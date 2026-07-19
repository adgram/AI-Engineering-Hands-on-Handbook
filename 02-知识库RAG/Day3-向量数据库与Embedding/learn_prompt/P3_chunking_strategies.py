from pathlib import Path
import re, json
from langchain_text_splitters import RecursiveCharacterTextSplitter

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "common" / "text_data"
_output_file = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

DATA_FILES = {
    "1": ("民用建筑设计统一标准.md", "markdown"),
    "2": ("enum.py", "python"),
    "3": ("excerpts.json", "json"),
}

def select_data():
    print("选择测试数据：")
    for k, (name, _) in DATA_FILES.items():
        fpath = _DATA_DIR / name
        size = fpath.stat().st_size
        print(f"  [{k}] {name} ({size//1000}KB)")
    choice = input("输入编号（默认 1）: ").strip() or "1"
    if choice not in DATA_FILES:
        print("无效选择，使用默认")
        choice = "1"
    fname, ftype = DATA_FILES[choice]
    fpath = _DATA_DIR / fname
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    if ftype == "json":
        data = json.loads(text)
        texts = []
        for item in data.get("excerpts", data if isinstance(data, list) else []):
            if isinstance(item, dict) and "content" in item:
                texts.append(item["content"])
        text = "\n\n".join(texts)
    return fname, text, ftype

def fixed_length_chunk(text: str, chunk_size: int = 200, overlap: int = 20) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def semantic_chunk(text: str) -> list:
    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    if len(sections) <= 1:
        sections = re.split(r'\n\s*\n', text)
    chunks = []
    for section in sections:
        if not section.strip():
            continue
        paragraphs = re.split(r'\n\s*\n', section.strip())
        for p in paragraphs:
            if len(p) > 50:
                chunks.append(p.strip())
            elif chunks:
                chunks[-1] += "\n" + p.strip()
    return chunks

def evaluate_chunking_strategy(text: str, chunk_func, label: str, **kwargs):
    chunks = chunk_func(text, **kwargs)
    result = []
    result.append(f"策略: {label}")
    result.append(f"参数: {kwargs}")
    result.append(f"块数: {len(chunks)}")
    avg = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    result.append(f"平均长度: {avg:.0f} 字")
    result.append(f"最短: {min(len(c) for c in chunks) if chunks else 0} 字")
    result.append(f"最长: {max(len(c) for c in chunks) if chunks else 0} 字")
    incomplete = sum(1 for c in chunks if not c.endswith(('。', '！', '？', '\n')))
    result.append(f"截断片段: {incomplete}/{len(chunks)}")
    result.append("")
    print("\n".join(result))
    with open(_output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(result) + "\n")
    return chunks

fname, text, ftype = select_data()

with open(_output_file, "w", encoding="utf-8") as f:
    f.write(f"数据源: {fname} ({ftype})\n")
    f.write(f"总字符: {len(text)}\n\n")

print(f"\n数据源: {fname}")
print(f"类型: {ftype}")
print(f"总字符: {len(text)}\n")

# 策略1：固定长度
evaluate_chunking_strategy(text, fixed_length_chunk, "固定长度", chunk_size=200, overlap=20)
evaluate_chunking_strategy(text, fixed_length_chunk, "固定长度(小窗口)", chunk_size=100, overlap=10)

# 策略2：递归字符（仅对非 markdown 时有效果差异）
if ftype in ("python", "json"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_text(text)
    result_lines = [f"策略: 递归字符切分", f"块数: {len(chunks)}"]
    result_lines.append("")
    with open(_output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(result_lines) + "\n")
    print("\n".join(result_lines))

# 策略3：语义切分（按标题/段落）
chunks_sem = evaluate_chunking_strategy(text, semantic_chunk, "语义切分")

# 展示前 3 块样例
with open(_output_file, "a", encoding="utf-8") as f:
    f.write("前3块样例:\n")
for i, chunk in enumerate(chunks_sem[:3]):
    info = f"块 {i+1} ({len(chunk)}字): {chunk[:80]}..."
    print(info)
    with open(_output_file, "a", encoding="utf-8") as f:
        f.write(info + "\n")

print(f"\n结果已写入 {_output_file}")
