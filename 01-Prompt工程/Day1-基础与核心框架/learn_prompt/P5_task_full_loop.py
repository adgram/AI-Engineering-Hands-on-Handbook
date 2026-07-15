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
    }
]

output = ""

# 第1步：模型决定调用函数
messages = [{"role": "user", "content": "北京今天天气怎么样？"}]
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

message = response.choices[0].message
reasoning_step1 = getattr(message, "reasoning_content", "")
messages.append(message)

output += f"=== 第1步：模型决定调用函数 ===\n"
output += f"思考过程: {reasoning_step1 or '（无显式思考过程）'}\n"
if message.tool_calls:
    for tc in message.tool_calls:
        output += f"调用: {tc.function.name}, 参数: {tc.function.arguments}\n"
output += "\n"

# 第2步：执行函数（模拟）
for tc in message.tool_calls:
    func_args = json.loads(tc.function.arguments)
    if tc.function.name == "get_weather":
        result = json.dumps({"city": func_args["city"], "temperature": 28, "condition": "晴", "humidity": "45%"})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        output += f"=== 第2步：函数返回 ===\n{result}\n\n"

# 第3步：将结果返回模型，生成最终回答
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=tools,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=500
)

reasoning_step3 = getattr(response.choices[0].message, "reasoning_content", "")
final_reply = response.choices[0].message.content

output += f"=== 第3步：模型最终回答 ===\n"
output += f"思考过程: {reasoning_step3 or '（无显式思考过程）'}\n"
output += f"回复: {final_reply}\n"

output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"

with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)

print(f"结果已写入 {output_file_name}")