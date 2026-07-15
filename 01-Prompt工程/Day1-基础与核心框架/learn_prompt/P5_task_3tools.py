import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式，如 123*456"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "将文本翻译为目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "待翻译的文本"},
                    "target": {"type": "string", "description": "目标语言，如 '英语'、'日语'"}},
                "required": ["text", "target"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "北京今天天气怎么样？顺便算一下 12345 × 6789，再把'hello world'翻译成中文"}],
    tools=tools,
    tool_choice="auto",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", "")

output = f"""思考过程: {reasoning or "（无显式思考过程）"}

"""
if message.tool_calls:
    for tc in message.tool_calls:
        output += f"调用: {tc.function.name}\n"
        output += f"参数: {tc.function.arguments}\n\n"
else:
    output += f"直接回复: {message.content}\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")