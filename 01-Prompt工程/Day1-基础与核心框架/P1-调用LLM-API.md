# P1：调用 LLM API — 第一次和 AI 对话

> **提示词工程（Prompt Engineering）** 是一门通过设计和优化输入提示词来引导大语言模型生成预期输出的技术。它位于"人-AI 协作"的最前沿——不需要训练模型，也不需要深厚的数学背景，只需学会如何与 AI 正确"沟通"。好的提示词可以让模型输出准确、可靠、可控；差的提示词则会让模型答非所问、产生幻觉。本部分将从调用 API 开始，逐步深入角色设定、思维链、结构化输出、模板化等工程化方法，用于掌握提示词工程的核心技能。

## 目标

成功调用 DeepSeek API，了解大模型输出数据的结构，理解思考模式开关、思考强度控制和 Token 基本概念

## 前置准备

- 已有 DeepSeek API Key
- Python 3.12 已安装
- 已安装 `pip install openai`

## 快速开始

### 1. 创建项目文件结构

下面是示例目录结构，其他示例可采用相同结构。

```
learn_prompt/
├── P1_llm.py          # 第一个 LLM 调用
├── P1_llm_result.txt  # 输出结果（自动生成）
```

### 2. 配置 API Key

创建 `.env` 文件：

```ini
DEEPSEEK_API_KEY=sk-your-actual-key-here
```

> 按照 .env.example 创建 .env 文件，并把 `sk-your-actual-key-here` 替换成真实 Key。然后已安装 `pip install python-dotenv`，使用 `load_dotenv()` 方式导入。

### 3. 编写第一个调用脚本

```python
from openai import OpenAI

client = OpenAI(
    api_key= DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "用一句话解释什么是 Large Language Model？"}
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
    max_tokens=200
)

print(response)
```

### 4. 运行

```bash
python P1_llm.py
```

输出内容如下

```markdown
ChatCompletion(id='1d362aef-6d4e-43a5-9126-3586c22337c5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='大型语言模型是一种通过海量文本数据训练而成的深度学习模型，能够理解和生成人类语言。', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content='我们要求用一句话解释什么是Large Language Model。需要简洁、准确。LLM是大型语言模型，基于大量文本数据训练，能生成自然语言文本。一句话：大型语言模型是一种通过海量文本数据训练而成的深度学习模型，能够理解和生成人类语言。'))], created=1783921143, model='deepseek-v4-flash', object='chat.completion', moderation=None, service_tier=None, system_fingerprint='fp_8b330d02d0_prod0820_fp8_kvcache_20260402', usage=CompletionUsage(completion_tokens=77, prompt_tokens=12, total_tokens=89, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=56, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cache_write_tokens=None, cached_tokens=0), prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=12))
```

## 思考模式控制

### 思考模式开关

DeepSeek V4 Flash 默认启用了思考模式（`thinking` 默认值为 `enabled`）。通过 `extra_body` 参数控制：

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "enabled"}},   # 启用思考模式
    # extra_body={"thinking": {"type": "disabled"}},  # 禁用思考模式
)
```

> 该参数必须通过 `extra_body` 传入（OpenAI SDK 格式）。

### 思考强度控制

`reasoning_effort` 控制模型思考的深度：

| 值 | 说明 |
|----|------|
| `"high"` | 标准思考强度（默认，适用于大部分请求） |
| `"max"` | 最大思考强度（适用于复杂 Agent 类请求如 Claude Code、OpenCode） |

注意：`low`、`medium` 会映射为 `high`，`xhigh` 会映射为 `max`。

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    reasoning_effort="high",                        # 思考强度
    extra_body={"thinking": {"type": "enabled"}},   # 思考模式开关
)
```

### 思考链输出

思考模式下，模型的思维链通过 `reasoning_content` 返回，与回复内容 `content` 同级：

```python
reasoning = getattr(response.choices[0].message, "reasoning_content", "")
content = response.choices[0].message.content
```

## 返回数据解析

上面打印的 `response` 是一个 `ChatCompletion` 对象，核心字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 请求的唯一标识 |
| `choices[].message.content` | `str` | 模型的最终回复 |
| `choices[].message.reasoning_content` | `str` | 模型的思考链（思考模式下有值） |
| `choices[].finish_reason` | `str` | 结束原因，`stop` 表示正常结束 |
| `model` | `str` | 实际使用的模型名称 |
| `usage.prompt_tokens` | `int` | 输入消耗的 Token 数 |
| `usage.completion_tokens` | `int` | 输出消耗的 Token 数 |
| `usage.total_tokens` | `int` | 总 Token 数（输入+输出） |
| `usage.completion_tokens_details.reasoning_tokens` | `int` | 思考过程消耗的 Token 数 |

其中 `reasoning_content`（思考链）和 `content`（最终回复）的关系为：

> **思考链**（`reasoning_content`）→ 模型内部推理过程 → **最终回复**（`content`）

注意：思考模式消耗的 Token 会计入 `completion_tokens`，在 `reasoning_tokens` 中单独列出。

## 采样参数（Temperature / Top-p）

Temperature 与 Top-p 是控制**生成随机性**的两个经典采样参数，主要用于非思考模式下的开放性生成：

- **Temperature（温度）**：对模型输出的概率分布做缩放。值越低（接近 0）输出越确定、越保守、越可复现；值越高输出越随机、越发散、越有创造性。
- **Top-p（核采样 / Nucleus Sampling）**：只从累计概率达到 `p` 的最小词集中采样，超过 `p` 的长尾词被截断。值越小输出越集中，越大越发散。

>  **注意：深度思考 / 推理模型时参数无效**：在 DeepSeek 思考模式、OpenAI o 系列等推理模型中，思考过程为保证推理正确性通常采用**确定性解码**，Temperature / Top-p **不生效**（或仅作用于最终总结部分）。因此本教程用 `reasoning_effort` 控制思考强度，而不依赖 Temperature / Top-p。若调用非思考模型，可再用 Temperature / Top-p 调节随机性。

## 核心概念

| 概念 | 说明 | 通俗理解 |
|------|------|----------|
| **System Prompt** | 系统级指令，设定模型角色和行为规则 | 给 AI 设定人设和规则 |
| **User Prompt** | 用户的输入/问题 | 用户提问 |
| **Assistant Response** | 模型的最终回复内容 | AI 的回答 |
| **Reasoning Content** | 模型的思考链（思考模式下返回） | AI 的内心独白 |
| **Token** | LLM 处理的最小单位（中文通常 1 字 ≈ 1-2 Token，取决于分词器） | LLM 看的"字" |
| **Max Tokens** | 限制总输出长度 | AI 最多说多少字 |
| **Reasoning Effort** | 思考强度控制（high/max） | AI 思考多用力 |

## 动手实验

### 实验 A：不同 System Prompt 对比

为`messages`字段添加`"system"	`角色，并对比不同角色的输出差异。

```python
# 角色1
messages=[
    {"role": "system", "content": "你是一个严肃的科学家"},
    {"role": "user", "content": "为什么天是蓝色的？"}
]
# 角色2
messages=[
    {"role": "system", "content": "你是一个诗人"},
    {"role": "user", "content": "为什么天是蓝色的？"}
]
```

**观察**：同一个问题，角色设定不同，回答风格完全不同。

### 实验 B：观察思考链

```python
reasoning = getattr(response.choices[0].message, "reasoning_content", "")
print("思考过程:", reasoning[:500])
print("---")
print("最终回答:", response.choices[0].message.content)
```

**观察**：模型的推理步骤清晰可见，可以看到它"怎么想的"再"怎么回答"。

### 实验 C：思考强度对比

```python
# 用同样的简单问题，分别用 high 和 max 跑一次
for effort in ["high", "max"]:
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "9.11 和 9.8 哪个大？"}],
        reasoning_effort=effort,
        extra_body={"thinking": {"type": "enabled"}}
    )
    reasoning = getattr(resp.choices[0].message, "reasoning_content", "")
    print(f"\n=== effort={effort} ===")
    print(f"思考链长度: {len(reasoning)} 字")
    print(f"回答: {resp.choices[0].message.content}")
```

**观察**：`max` 的思考链比 `high` 更长，但同时也消耗更多 Token。

### 实验 D：Token 计数

```python
print(f"输入 Token: {response.usage.prompt_tokens}")
print(f"输出 Token: {response.usage.completion_tokens}")
print(f"思考链字数: {len(reasoning)}")
print(f"回答字数: {len(content)}")
```

**观察**：思考链会消耗大量输出 Token，思考模型的总 Token 消耗高于非思考模式。

## 完成标准

- [ ] 成功调用 API，看到思考过程和最终回复
- [ ] 理解思考模式开关（`thinking.enabled`）的作用
- [ ] 理解思考强度（`reasoning_effort`）的区别
- [ ] 知道 Temperature / Top-p 在思考模式下不生效
- [ ] 能区分 Reasoning Content 和 Content

## 下一步
完成 → 进入 [P2-角色设定与消息结构](P2-角色设定与消息结构.md)

