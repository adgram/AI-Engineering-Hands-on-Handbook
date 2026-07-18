# Prompt工程 | P1：调用 LLM API 和 AI 对话

## 前言

我们平时使用AI，要么是通过网页端直接交互，要么是通过工具调用API。这些工具不仅在调用AI接口，还在调用时附加一些辅助信息、工具指令、历史对话等额外内容。这些信息帮助AI理解，给予AI性格，增强AI能力，约束AI行为，赋予AI记忆力的附加信息就是**提示词**。

现在**提示词**已经成为了一门学问。好的提示词可以让模型输出准确、可靠、可控；差的提示词则会让模型答非所问、产生幻觉。**提示词编写**也是API调用的基本功之一。

本部分是《AI应用工程实战——Prompt / RAG / Agent》系列的学习笔记，本节是**提示词Prompt**部分的第一节。该部分的完整教程和演示代码详见：https://github.com/adgram/AI-Engineering-Hands-on-Handbook。

## 准备

本文采用`deepseek-v4-flash`进行演示，工具则使用 `Python`的`openai`库。

本节的目标是成功调用 DeepSeek API，了解大模型输出数据的结构，理解思考模式开关、思考强度控制和 Token 基本概念

## 快速开始

下面是最简单的调用方式：

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

输出内容如下：

```markdown
ChatCompletion(id='1d362aef-6d4e-43a5-9126-3586c22337c5', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='大型语言模型是一种通过海量文本数据训练而成的深度学习模型，能够理解和生成人类语言。', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, reasoning_content='我们要求用一句话解释什么是Large Language Model。需要简洁、准确。LLM是大型语言模型，基于大量文本数据训练，能生成自然语言文本。一句话：大型语言模型是一种通过海量文本数据训练而成的深度学习模型，能够理解和生成人类语言。'))], created=1783921143, model='deepseek-v4-flash', object='chat.completion', moderation=None, service_tier=None, system_fingerprint='fp_8b330d02d0_prod0820_fp8_kvcache_20260402', usage=CompletionUsage(completion_tokens=77, prompt_tokens=12, total_tokens=89, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=56, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cache_write_tokens=None, cached_tokens=0), prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=12))
```

### 思考模式控制

DeepSeek V4 Flash 默认启用了思考模式（`thinking` 默认值为 `enabled`）。通过 `extra_body` 参数控制：

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "enabled"}},   # 启用思考模式
    # extra_body={"thinking": {"type": "disabled"}},  # 禁用思考模式
)
```

`reasoning_effort` 控制模型思考的深度：

| 值 | 说明 |
|----|------|
| `"high"` | 标准思考强度（默认，适用于大部分请求） |
| `"max"` | 最大思考强度（适用于复杂 Agent 类请求如 Claude Code、OpenCode） |

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

**思考链**（`reasoning_content`）→ 模型内部推理过程 → **最终回复**（`content`）

注意：思考模式消耗的 Token 会计入 `completion_tokens`，在 `reasoning_tokens` 中单独列出。

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

## 案例

### 例1：不同 System Prompt 对比

通过为`messages`字段添加`"system"	`角色，可以看出同一个问题，角色设定不同，回答风格完全不同。

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

### 例1：获取思维链

下面方式可以获取AI的推理步骤

```python
reasoning = getattr(response.choices[0].message, "reasoning_content", "")
print("思考过程:", reasoning[:500])
print("---")
print("最终回答:", response.choices[0].message.content)
```
