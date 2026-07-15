import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

messages = [
    {"role": "system", "content": "你是一个逻辑推理专家，回答前先一步步推理"},
    {"role": "user", "content": "所有猫都怕水。Tom 是一只猫。Tom 怕水吗？"},
    {"role": "assistant", "content": "前提1: 所有猫都怕水。前提2: Tom 是一只猫。根据前提1 和前提2，Tom 是所有猫中的一员，所以 Tom 也怕水。结论: 是的，Tom 怕水。"},
    {"role": "user", "content": "所有会飞的动物都有翅膀。蝙蝠会飞。企鹅是鸟类但不会飞。问题：\n1. 蝙蝠有翅膀吗？\n2. 企鹅是鸟吗？\n3. 所有鸟类都会飞吗？"}
]

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
content = resp.choices[0].message.content

output = f"""思考过程: {reasoning or "（无显式思考过程）"}

回复: {content}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")