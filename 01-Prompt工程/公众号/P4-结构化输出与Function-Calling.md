# Prompt工程 | P4：结构化输出与 Function Calling

## 准备

可以对比上一节，本节主要内容是，采用更严格的方式让AI输出标准的结构化数据。本节包含JSON数据输出和工具调用两部分。

## 常用结构化格式对比

下面是常用的一些输出格式，均相对简单，其他复杂的格式则通常需要调用工具辅助：

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **JSON** | 可直接 `json.loads()` 解析，通用性强 | 语法严格，字段缺失/类型不匹配易报错 | API 数据交换、信息提取 |
| **XML** | 层级清晰，可读性好，支持属性 | 体积较大，解析比 JSON 重 | 复杂嵌套数据、文档标记 |
| **YAML** | 流式友好，体积小，可读性好 | 对缩进敏感，解析可能不唯一 | 配置文件、元数据 |
| **Markdown** | 可读性最好，展示友好 | 程序解析复杂，无标准 Schema | 文档生成、报告输出 |

### 方法一：Prompt 引导 JSON 输出

通过 System 指令 + Few-shot 示例**引导**模型输出 JSON：

```python
messages = [
    {"role": "system", "content": "你是一个信息提取助手，只输出 JSON 格式"},
    {"role": "user", "content": "张三，35岁，毕业于北京大学计算机系，现为阿里巴巴算法工程师"},
    {"role": "assistant", "content": '{"name": "张三", "age": 35, "education": {"school": "北京大学", "major": "计算机系"}, "job": {"company": "阿里巴巴", "position": "算法工程师"}}'},
    {"role": "user", "content": "李四，28岁，上海交通大学硕士，在字节跳动做产品经理，擅长数据分析"}
]
```

### 方法二：JSON Mode（response_format）

DeepSeek 支持使用 `response_format={"type": "json_object"}` 强制 JSON 输出：

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "提取以下文本中的信息，输出 JSON 格式"},
        {"role": "user", "content": "昨天下午3点，李明在北京朝阳区建国路88号丢失了黑色钱包"},
    ],
    response_format={"type": "json_object"},  # 强制 JSON
)
```

### 技巧：用 JSON Mode 做推理（Inference Engine）

使用 System 角色规定格式，然后使用 response_format 强制输出 json。

情感分析、主题识别、意图识别是结构化输出的典型用法。一次 API 调用替代多条规则引擎 + 多个分类器，维护成本更低：

```python
client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "分析用户评论，按以下 json 结构输出：\n{ \"sentiment\": \"正面/负面/neutral/mixed\", \"topics\": [\"主题1\", \"主题2\"], \"intent\": \"抱怨/咨询/赞美/对比\", \"urgency\": \"high/medium/low\" }"},
        {"role": "user", "content": "这个手机电池太不耐用了，半天就没电，但拍照效果还不错"},
    ],
    response_format={"type": "json_object"},
    max_tokens=500,
)
```

注意：Prompt 中必须包含 `json` 字样并给出期望结构，否则 JSON Mode 可能不会生效；设置 `max_tokens` 防止 JSON 被截断；对空 content 做容错处理。

- JSON Mode 注意事项（来自 DeepSeek 官方文档）

| 规则 | 说明 |
|------|------|
| Prompt 必须含 "json" | System 或 User Prompt 中必须出现 `json` 字样，并给出期望的 JSON 格式样例 |
| 设 max_tokens | 防止 JSON 字符串被截断导致解析失败 |
| 可能返回空 content | 有概率返回空的 `content`，建议在代码中做容错处理 |

### 原生结构化输出（json_schema）

除了用 Prompt 引导格式，现在很多模型也支持原生结构化输出——将 Schema 作为 API 参数传入，由模型服务层做格式约束，比纯自然语言要求更可靠：

```python
# 示意：将 Schema 声明在请求中（具体 API 因厂商而异）
response = client.chat.completions.create(
     model="...",
     messages=[...],
     response_format={"type": "json_schema", "schema": {...}}  # JSON Schema
)
```

不同厂商的实现不同，需做本地校验和失败重试。换模型/换 SDK 时建议先跑兼容性测试。

> 注意：原生结构化输出依赖模型和框架支持。建议准备降级策略——解析失败时记录日志、触发重试或给默认值兜底。

## Function Calling（工具调用）

通过 `tools` 参数注册函数描述，可以让模型自主判断何时调用哪个函数，并提取参数，这是 Agent 的基础能力。

`tools` 参数（请求时传入）定义模型可以调用的工具列表；`tool` 角色（对话消息）用于把工具执行结果送回模型。两者配合完成一次工具调用：定义 → 模型申请调用 → 本地调用 → 返回结果 → 模型根据调用结果进行回复。

### 工具定义结构

每个工具包含三个核心字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `type` | 固定为 `"function"` | `"function"` |
| `function.name` | 函数名，模型通过此名称选择调用 | `"get_weather"` |
| `function.description` | 函数功能描述，越清晰模型判断越准 | `"获取指定城市的天气信息"` |
| `function.parameters` | 参数的 JSON Schema 定义 | 见下 |

参数（`parameters`）使用 JSON Schema 格式：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如：北京、上海"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-MM-DD"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

`required` 标注必填参数，未在 `required` 中的参数模型可省略。

### 请求参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `tools` | 函数定义列表 | 定义好的工具列表 |
| `tool_choice` | 控制模型何时调用工具 | `"auto"`（默认，自主判断）/ `"required"`（必须调一个）/ `"none"`（禁止调用）/ `{"type":"function","function":{"name":"xxx"}}`（强制调指定函数）|

### 响应解析

模型若决定调用函数，返回的 `message` 会包含 `tool_calls`：

```python
message = response.choices[0].message
if message.tool_calls:
    for tc in message.tool_calls:
        print(f"函数: {tc.function.name}")
        print(f"参数: {tc.function.arguments}")  # JSON 字符串
        print(f"调用ID: {tc.id}")  # 用于 tool 角色回传
```

### 完整流程

```
用户输入 → 模型判断意图 → 输出函数名+参数（tool_calls）
   → 本地执行函数 → 用 tool 角色返回结果
   → 模型基于结果生成最终回答
```

## 容错模式

生产环境中 JSON 解析和函数调用的失败场景应有标准化的恢复策略：

| 失败类型 | 现象 | 恢复策略 |
|----------|------|----------|
| JSON 解析失败 | `json.loads()` 抛异常 | 重新请求，增加格式约束；重试 2 次后降级为默认值 |
| 空 content | JSON Mode 返回空字符串 | 检查 Prompt 是否包含 "json" 关键词；返回兜底默认值 |
| 字段缺失 | 模型遗漏某些必填字段 | 在 Prompt 中标注 required；用 `dict.get()` 设置默认值 |
| 类型错误 | 字符串给了 number 字段 | 在参数描述中加 `minimum`/`maximum`/`enum` 约束 |
| Tool Call 400 | 思考模式下缺 `reasoning_content` | 始终用 `messages.append(message)` 而非手动构造 |

```python
def safe_json_parse(content: str, default: dict = None) -> dict:
    if not content:
        print("警告：空 content，返回默认值")
        return default or {}
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}，尝试修复...")
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return default or {}
```

### 思考模式下的 Tool Calls 注意事项

当前使用 `deepseek-v4-flash`，当进行工具调用时需注意：

- **带工具调用的轮次，`reasoning_content` 必须回传给 API**，否则会报 400 错误
- 使用 `messages.append(response.choices[0].message)` 追加整条消息（含 `reasoning_content`、`tool_calls`），框架会自动处理
- 不要手动构造 assistant 消息，始终追加完整 message 对象

## 案例

### 例1：简历信息提取（JSON 输出）

```python
messages = [
    {"role": "system", "content": "你是一个简历解析助手，只输出 JSON。提取姓名、学历、技能、工作经历"},
    {"role": "user", "content": "我叫王磊，2018年毕业于华中科技大学计算机硕士。精通 Python、Java、Docker。先在字节跳动做了3年后端开发，现在在蚂蚁集团做技术架构师。"},
    {"role": "assistant", "content": '{"name": "王磊", "education": {"school": "华中科技大学", "degree": "硕士", "major": "计算机科学与技术"}, "skills": ["Python", "Java", "Docker"], "experience": [{"company": "字节跳动", "position": "后端开发", "years": 3}, {"company": "蚂蚁集团", "position": "架构师"}]}'},
    {"role": "user", "content": "我是张薇，本科毕业于复旦新闻系，有5年新媒体运营经验。熟悉PS、PR、公众号排版。现在是一家MCN机构的内容主管。"}
]
```

### 例2：定义 3 个工具，让模型自主选择

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
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
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 123*456"}
                },
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
                    "target": {"type": "string", "description": "目标语言，如 '英语'、'日语'"}
                },
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
)
```

### 例3：完整循环（调用函数→返回结果→模型最终回答）

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

# 第1步：模型决定调用函数
messages = [{"role": "user", "content": "北京今天天气怎么样？"}]
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message
messages.append(message)  # 追加模型的 tool_calls 消息

# 第2步：执行函数（模拟）
for tc in message.tool_calls:
    func_args = json.loads(tc.function.arguments)
    if tc.function.name == "get_weather":
        # 模拟天气 API 返回
        result = json.dumps({"city": func_args["city"], "temperature": 28, "condition": "晴", "humidity": "45%"})
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result
        })

# 第3步：将结果返回模型，生成最终回答
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    tools=tools,
)
```
