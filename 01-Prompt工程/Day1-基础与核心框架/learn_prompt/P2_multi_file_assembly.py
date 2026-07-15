import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

PROMPT_DIR = Path(__file__).parent / "prompt_files"


def load_prompt_file(filename):
    return open(PROMPT_DIR / filename, encoding="utf-8").read()


def assemble_system_prompt():
    sections = {
        "角色与使命": f"你是「文笔 AI」\n{load_prompt_file('SOUL.md')}",
        "行为准则与工作流程": load_prompt_file("AGENTS.md"),
        "知识边界": load_prompt_file("MEMORY.md"),
        "可执行操作": load_prompt_file("TOOLS.md"),
    }
    return "\n\n".join(
        f"## {k}\n{v}" for k, v in sections.items()
    )


simple_prompt = "你是一个中文写作助手"
file_prompt = assemble_system_prompt()

results = []
for label, sp in [("单行 Prompt", simple_prompt), ("多文件组装", file_prompt)]:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": sp},
            {"role": "user", "content": "请帮我润色这段话：今天天气很好，所以我们决定去公园玩。公园里有很多人，有的在跑步，有的在放风筝，还有的在野餐。"}
        ],
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=500
    )
    reasoning = getattr(response.choices[0].message, "reasoning_content", "")
    content = response.choices[0].message.content
    results.append(f"=== {label} ===\n思考过程: {reasoning or '（无显式思考过程）'}\n\n回复: {content}")

output = "\n\n".join(results)

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
