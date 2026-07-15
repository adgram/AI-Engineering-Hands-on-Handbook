import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

resp_knowledge = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "关于金门大桥的维护，列举几个关键事实"}],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

knowledge = resp_knowledge.choices[0].message.content
knowledge_reasoning = getattr(resp_knowledge.choices[0].message, "reasoning_content", "")

resp_answer = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "利用以下知识回答问题"},
        {"role": "user", "content": f"知识：\n{knowledge}\n\n问题：金门大桥为什么是橙色的？"}
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

answer_reasoning = getattr(resp_answer.choices[0].message, "reasoning_content", "")
answer = resp_answer.choices[0].message.content

output = f"""=== 第一步：生成知识 ===
思考过程: {knowledge_reasoning or "（无显式思考过程）"}

生成的知识: {knowledge}

=== 第二步：基于知识回答问题 ===
思考过程: {answer_reasoning or "（无显式思考过程）"}

最终回答: {answer}
"""

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")