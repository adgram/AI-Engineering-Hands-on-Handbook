import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent/"common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient
client = LLMClient()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市"},
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "搜索酒店信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市"},
                    "max_price": {"type": "number", "description": "最高价格/晚"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "预订机票",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "出发城市"},
                    "to": {"type": "string", "description": "到达城市"},
                    "date": {"type": "string", "description": "出发日期 YYYY-MM-DD"}
                },
                "required": ["from", "to", "date"]
            }
        }
    }
]

def execute_function(name, args):
    if name == "get_weather":
        return json.dumps({"city": args["city"], "date": args.get("date"), "temperature": "22°C", "condition": "晴"})
    elif name == "search_hotels":
        return json.dumps({"city": args["city"], "results": [
            {"name": "和平饭店", "price": 1200, "stars": 5, "rating": 4.8},
            {"name": "华尔道夫", "price": 1500, "stars": 5, "rating": 4.9}
        ]})
    elif name == "book_flight":
        return json.dumps({"from": args["from"], "to": args["to"], "date": args["date"], "price": 850, "available": True})
    return json.dumps({"error": "unknown function"})

current_date = "2026年7月15日，星期三"
messages = [
    {"role": "system", "content": f"今天是{current_date}。根据这个日期计算出下周三的具体日期，调用对应工具。"},
    {"role": "user", "content": "帮我查下周三北京飞上海的机票，再查一下上海当天天气"}
]
output_lines = ["开始多轮函数调用...\n"]

max_rounds = 5
for round_num in range(max_rounds):
    response = client.chat(
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    message = response.choices[0].message

    if not message.tool_calls:
        output_lines.append(f"\n最终回答: {message.content}")
        break

    messages.append(message)

    for tc in message.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        output_lines.append(f"  [第{round_num+1}轮] 调用: {func_name}({func_args})")

        result = execute_function(func_name, func_args)
        output_lines.append(f"  [第{round_num+1}轮] 结果: {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result
        })

output = "\n".join(output_lines)
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")