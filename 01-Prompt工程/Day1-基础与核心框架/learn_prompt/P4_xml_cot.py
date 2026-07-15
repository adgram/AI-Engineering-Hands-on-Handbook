import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

xml_prompt = """
<question>
小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
</question>

<reasoning>
逐步推理：
</reasoning>
<answer>
</answer>
"""

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": f"填充 `<reasoning>` 和 `<answer>` 标签的内容，输出结构固定。{xml_prompt}"}],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
content = resp.choices[0].message.content

base_name = Path(__file__).parent / Path(__file__).stem

with open(f"{base_name}_result.txt", "w", encoding="utf-8") as f:
    f.write(f"""思考过程: {reasoning or "（无显式思考过程）"}

回复: {content}""")

with open(f"{base_name}_result.xml", "w", encoding="utf-8") as f:
    f.write(content)

print(f"结果已写入 {base_name}_result.txt 和 {base_name}_result.xml")