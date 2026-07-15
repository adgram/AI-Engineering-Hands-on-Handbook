import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

positive_only = [
    {"role": "system", "content": "你是一个健康饮食顾问"},
    {"role": "user", "content": "请提供一些关于健康饮食的建议"}
]

with_constraints = [
    {"role": "system", "content": "你是一个健康饮食顾问"},
    {"role": "user", "content": """请提供关于健康饮食的建议，但：
- 不要包含具体的减肥建议
- 不要推荐特定品牌的产品
- 避免使用医学术语
- 重点关注日常可实施的饮食习惯"""}
]

output = ""
for label, msgs in [("只说正面", positive_only), ("正面+负面约束", with_constraints)]:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=msgs,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        max_tokens=500
    )

    reasoning = getattr(response.choices[0].message, "reasoning_content", "")
    content = response.choices[0].message.content

    output += f"=== {label} ===\n"
    if reasoning:
        output += f"思考过程: {reasoning}\n\n"
    output += f"{content}\n\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")