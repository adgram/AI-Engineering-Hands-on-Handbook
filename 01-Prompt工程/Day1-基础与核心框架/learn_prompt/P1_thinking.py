import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

output = ""

# 用同样的简单问题，分别用 high 和 max 跑一次
for effort in ["high", "max"]:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "9.11 和 9.8 哪个大？"}],
        reasoning_effort=effort,
        extra_body={"thinking": {"type": "enabled"}}
    )
    reasoning = getattr(response.choices[0].message, "reasoning_content", "")
    output += f"=== effort={effort} ===\n"
    output += f"思考链长度: {len(reasoning)} 字\n"
    output += f"思考过程:{reasoning}\n"
    output += f"回答: {response.choices[0].message.content}\n"
    output += f"输入 Token: {response.usage.prompt_tokens}\n"
    output += f"输出 Token: {response.usage.completion_tokens}\n\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")