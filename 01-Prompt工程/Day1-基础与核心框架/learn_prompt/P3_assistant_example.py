import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 不加示例：模型自由发挥
messages_no_example = [
    {"role": "system", "content": "你是一个会议纪要助手"},
    {"role": "user", "content": """整理以下会议记录：
参会人：张总、李工、王经理
内容：讨论了 Q3 预算审批，决定追加 20 万用于服务器升级。
李工提出前端重构方案，预计 8 月底完成。
王经理建议下周一前各团队提交进度报告。"""}
]

# 加示例：让模型学会结构化格式
messages_with_example = [
    {"role": "system", "content": "你是一个会议纪要助手"},
    {"role": "user", "content": """整理以下会议记录：
参会人：王总、刘工
内容：讨论了登录页改版方案，决定采用方案 B。
刘工建议周四前出原型图。"""},
    {"role": "assistant", "content": """## 议题：登录页改版
- **决定**：采用方案 B
- **待办**：刘工周四前出原型图"""},
    {"role": "user", "content": """整理以下会议记录：
参会人：张总、李工、王经理
内容：讨论了 Q3 预算审批，决定追加 20 万用于服务器升级。
李工提出前端重构方案，预计 8 月底完成。
王经理建议下周一前各团队提交进度报告。"""}
]

resp_no = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages_no_example,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=300
)

resp_yes = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages_with_example,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=300
)

reasoning_no = getattr(resp_no.choices[0].message, "reasoning_content", "")
content_no = resp_no.choices[0].message.content

reasoning_yes = getattr(resp_yes.choices[0].message, "reasoning_content", "")
content_yes = resp_yes.choices[0].message.content

output = f"""=== 不加示例 ===
思考过程: {reasoning_no or "（无显式思考过程）"}

回复: {content_no}

=== 加示例 ===
思考过程: {reasoning_yes or "（无显式思考过程）"}

回复: {content_yes}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")
