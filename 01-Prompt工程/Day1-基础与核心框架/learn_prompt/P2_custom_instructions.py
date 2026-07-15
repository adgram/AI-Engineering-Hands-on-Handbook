import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


class UserProfile:
    def __init__(self, role_desc: str, style_desc: str):
        self.role_desc = role_desc
        self.style_desc = style_desc

    def build_system_prompt(self) -> str:
        return f"""## 关于用户
{self.role_desc}

## 回答要求
{self.style_desc}"""


dev_profile = UserProfile(
    role_desc="我是一名有5年经验的前端开发者，主力 Vue 3 + TypeScript，使用 Vite 构建工具",
    style_desc="请以资深技术专家口吻回应。提供代码示例时用最新语法标准。用 Markdown 格式化代码块。"
)

messages = [
    {"role": "system", "content": dev_profile.build_system_prompt()},
    {"role": "user", "content": "请帮我制作一个简单的算术题应用"}
]

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=10000
)

reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content

output = f"""思考过程: {reasoning or "（无显式思考过程）"}

回复: {content}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")