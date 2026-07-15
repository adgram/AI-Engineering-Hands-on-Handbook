import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

simple_prompt = "你是一个营销文案写手"

costar_prompt = """
## Context（上下文）
我们是一家面向 25-35 岁年轻人的咖啡品牌，即将推出一款冷萃咖啡液。

## Objective（目标）
撰写一篇小红书种草文案，提升新品知名度。

## Style（风格）
轻松活泼，带个人体验感，使用 emoji 和话题标签。

## Tone（语气）
亲切、真诚，像朋友推荐一样自然。

## Audience（受众）
喜欢尝试新事物的年轻白领和大学生。

## Response（回应格式）
输出一段 200 字左右的文案，包含标题 + 正文 + 3 个标签。
"""

question = "帮我写一下这款冷萃咖啡液的种草文案"

output = ""
for label, prompt in [("普通 Prompt", simple_prompt), ("CO-STAR 框架", costar_prompt)]:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
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