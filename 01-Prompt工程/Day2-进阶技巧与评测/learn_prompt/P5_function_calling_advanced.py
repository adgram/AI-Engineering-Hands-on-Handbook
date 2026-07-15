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
            "name": "search_hotels",
            "description": "搜索酒店信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市"},
                    "check_in": {"type": "string", "description": "入住日期 YYYY-MM-DD"},
                    "check_out": {"type": "string", "description": "离店日期 YYYY-MM-DD"},
                    "stars": {"type": "integer", "description": "酒店星级（3/4/5）", "minimum": 1, "maximum": 5},
                    "max_price": {"type": "number", "description": "最高价格/晚"},
                    "amenities": {"type": "array", "items": {"type": "string"}, "description": "设施要求，如 wifi/泳池/健身房"}
                },
                "required": ["city", "check_in", "check_out"]
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
                    "date": {"type": "string", "description": "出发日期 YYYY-MM-DD"},
                    "passengers": {"type": "integer", "description": "乘客数量"},
                    "cabin_class": {"type": "string", "enum": ["economy", "business", "first"]}
                },
                "required": ["from", "to", "date"]
            }
        }
    },
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
    }
]

response = client.chat(
    messages=[{"role": "user", "content": "帮我查下周三北京飞上海的机票，再查一下上海当天天气，顺便找找上海外滩附近的五星酒店"}],
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message
output_lines = []
if message.tool_calls:
    output_lines.append(f"模型选择了 {len(message.tool_calls)} 个函数:")
    for tc in message.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        output_lines.append(f"  🔧 {func_name}({json.dumps(func_args, ensure_ascii=False)})")
else:
    output_lines.append(f"模型直接回复: {message.content}")

output = "\n".join(output_lines)
output_file_name = Path(__file__).parent / f"{Path(__file__).stem}_result.txt"
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(output)
print(f"结果已写入 {output_file_name}")