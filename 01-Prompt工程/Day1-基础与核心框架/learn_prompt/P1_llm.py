import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "用一句话解释什么是 Large Language Model？"}
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=200
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(str(response))

print(f"结果已写入 {output_file_name}")