import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


class PromptTemplates:
    @staticmethod
    def qa(context: str, question: str) -> list:
        system = "你是一个知识库助手，请基于以下上下文回答问题。如果上下文不足以回答，请说'根据已有信息无法回答'。"
        user = f"""上下文：
{context}

问题：
{question}

请基于上下文回答："""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    @staticmethod
    def extract_info(text: str, fields: list) -> list:
        fields_str = "\n".join([f"- {f}" for f in fields])
        system = "你是一个信息提取专家，只输出 JSON。"
        user = f"""从以下文本中提取信息，输出 JSON 格式，包含以下字段：
{fields_str}

文本：{text}"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    @staticmethod
    def translate(text: str, target_lang: str, style: str = "正式") -> list:
        return [
            {"role": "system", "content": f"你是一个专业翻译，将文本翻译为{target_lang}，风格：{style}"},
            {"role": "user", "content": text}
        ]


messages = PromptTemplates.qa(
    context="检索增强生成（Retrieval-augmented Generation），简称RAG，是当下热门的大模型前沿技术之一。\n检索增强生成模型结合了语言模型和信息检索技术。具体来说，当模型需要生成文本或者回答问题时，它会先从一个庞大的文档集合中检索出相关的信息，然后利用这些检索到的信息来指导文本的生成，从而提高预测的质量和准确性。",
    question="RAG 全称是什么？"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
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