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
        {"role": "system", "content": "你是一个诗人"},
        {"role": "user", "content": "为什么天是蓝色的？"}
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "disabled"}},
    max_tokens=200
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content

output = f"""思考过程:
{reasoning or "（无显式思考过程）"}

---

模型回复:
{content}

---

Token 用量: {response.usage}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")