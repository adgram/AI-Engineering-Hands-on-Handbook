# P5：Function Calling 进阶 — 多函数、并行调用

## 目标
掌握多函数定义、并行函数调用（Parallel Tool Calls），为 Agent 编排做准备

> **定位**：Day1 P5 讲解了 Function Calling 基础（单函数定义、JSON Schema、完整循环），本篇在其之上扩展为多函数、并行调用和 PAL。本篇的多函数循环和 tool_choice 策略是 Agent 阶段 ToolRegistry 和 ReAct 循环的基础。

## 回顾 Day1 P5

Day1 P5 中 模型根据用户输入，输出函数名和参数，由开发者执行。但实践中用户往往同时需要多个服务（查天气 + 订酒店 + 订机票），这就需要并行调用支持。

本讲的核心问题：**如何让模型在单次请求中同时决定调用多个函数，并把结果反馈给模型以生成最终答案？**

## 1：多函数 + 复杂参数

定义多个工具（酒店搜索、机票预订、天气查询），函数参数支持嵌套对象、枚举值和数组。模型会自动匹配用户意图并决定调用哪个函数、传什么参数，一次请求内可以并行调用多个互不依赖的工具。

```python
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

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "帮我查下周三北京飞上海的机票，再查一下上海当天天气，顺便找找上海外滩附近的五星酒店"}],
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message
if message.tool_calls:
    print(f"模型选择了 {len(message.tool_calls)} 个函数:")
    for tc in message.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        print(f"  🔧 {func_name}({json.dumps(func_args, ensure_ascii=False)})")
else:
    print(f"模型直接回复: {message.content}")
```

## 2：完整 Tool Call 循环

上面只展示了模型一次调用的结果。实际应用中需要「模型调用工具 → 执行 → 结果送回模型 → 模型综合回答」的多轮循环。DeepSeek V4 支持在一次响应中返回多个工具调用，开发者遍历执行后将结果以 `role: "tool"` 追加到消息列表，再发给模型继续推理。

```python
# 模拟执行函数
def execute_function(name, args):
    """模拟执行工具函数并返回结果"""
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

# 完整调用循环
messages = [{"role": "user", "content": "帮我查下周三北京飞上海的机票，再查一下上海当天天气"}]
print("开始多轮函数调用...\n")

max_rounds = 5
for round_num in range(max_rounds):
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    message = response.choices[0].message
    
    if not message.tool_calls:
        # 没有函数调用，直接输出最终答案
        print(f"\n最终回答: {message.content}")
        break

    # 添加模型回复（包含工具调用请求和 reasoning_content）
    messages.append(message)
    
    # 处理每个工具调用
    for tc in message.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        print(f"  [第{round_num+1}轮] 调用: {func_name}({func_args})")
        
        # 执行函数
        result = execute_function(func_name, func_args)
        print(f"  [第{round_num+1}轮] 结果: {result}")
        
        # 将函数结果返回给模型
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result
        })
```

> 注意：思考模式下，带工具调用的轮次必须使用 `messages.append(message)`（完整 message 对象含 `reasoning_content`），不能手动构造 assistant 消息，否则后续请求会 400 报错。

执行流程示意：

```
用户提问
   ↓
模型返回工具调用（可能多个）
   ↓
逐个执行函数 → 结果包装为 role:"tool"
   ↓
追加到消息列表，再次请求模型
   ↓
模型综合所有结果输出最终回答
```

## 3：重要参数详解

控制模型是否调用工具、以及如何调用。选择合适的策略可以避免模型在不必要时浪费 token 或遗漏关键工具调用。

### tool_choice

| 值 | 行为 | 适用场景 |
|----|------|----------|
| `"auto"` | 模型自主决定是否调用工具 | 通用场景 |
| `"none"` | 不调用任何工具 | 纯文本对话 |
| `"required"` | 强制调用一个工具 | 必须使用工具的任务 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定工具 | 固定流程 |

### parallel_tool_calls

```python
# 默认启用（parallel_tool_calls=True）
# 模型可以在一次请求中同时调用多个工具
# 比如同时查天气 + 查酒店 + 订机票

# 禁用（parallel_tool_calls=False）
# 模型每次只调用一个工具，逐个执行
```

## 4：PAL（Program-Aided Language Model）

让模型写代码来代替推理，适用于精确计算类任务：

```
你有一台计算器可用，请把问题写成 Python 代码来求解。

问题：一个农场里有鸡和兔子共 35 只，脚共 94 只，鸡和兔子各多少只？

请输出可执行的 Python 代码，并给出运行结果。
```

与 Function Calling 的区别：

| 维度 | Function Calling | PAL |
|------|-----------------|-----|
| 思路 | 模型决定调哪个函数 | 模型自己写代码来解决问题 |
| 灵活性 | 受限于预定义的工具集 | 无限灵活（模型可以写任何代码） |
| 可靠性 | 参数验证容易 | 代码可能包含 Bug |
| 适用场景 | 固定操作（查天气、搜数据库） | 计算、逻辑、数据处理 |

```python
# PAL 模式的完整流程
# 1. 模型写代码
resp_code = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": pal_prompt}]
)
code = resp_code.choices[0].message.content

# 2. 提取代码部分执行
import re
code_block = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
if code_block:
    exec(code_block.group(1))  # 生产环境请用沙箱执行
```

> PAL（Program-Aided Language Model）适合精确计算的场景（数学、统计、数据处理）。缺点是需要自己处理代码提取和安全执行。

## 动手实验

1. 定义 5 个工具函数（搜索、计算、翻译、生成图片、发送邮件），让模型根据复杂需求自动选择
2. 实现一个带错误处理的工具调用循环（工具调用失败 → 重试或报错）
3. 对比 `parallel_tool_calls=True` 和 `False` 的行为差异
4. 在思考模式下运行工具循环，故意不传 `reasoning_content` 看是否报 400 错误

## 完成标准
- [ ] 能定义多函数、带复杂参数的工具
- [ ] 实现了完整的「模型选择 → 执行 → 返回 → 最终回答」循环
- [ ] 理解 `tool_choice` 各个值的区别
- [ ] 能处理工具调用失败的情况

## 下一步 → [P6-Harness工程](P6-Harness工程.md)
