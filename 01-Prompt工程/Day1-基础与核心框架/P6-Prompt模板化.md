# P6：Prompt 模板化

## 目标
学会用模板管理 Prompt，让代码更干净、可复用，为后续 RAG 和 Agent 打下基础

## 为什么需要模板？

```
原始方式（硬编码）：
prompt = f"你是一个{role}，请用{style}风格回答用户的问题：{question}"

推荐做法（模板方式）：
template = "你是一个{role}，请用{style}风格回答用户的问题：{question}"
prompt = template.format(role="医生", style="专业严谨", question="头痛怎么办")
```

当 Prompt 超过 20 行、有多个变量、多种角色时，模板的优势非常明显。

## 方法一：Python f-string / str.format（最轻量）

对于存在固定对话模式的场景，制作简单固定格式提示模板，预留上行文和用户输入位置，在新对话中动态填充：

```python
# prompt_templates.py

class PromptTemplates:
    """集中管理所有 Prompt 模板"""
    @staticmethod
    def qa(context: str, question: str) -> list:
        """RAG 问答模板"""
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
        """信息提取模板"""
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
        """翻译模板"""
        return [
            {"role": "system", "content": f"你是一个专业翻译，将文本翻译为{target_lang}，风格：{style}"},
            {"role": "user", "content": text}
        ]
```

> 该方法配合类型注解可做参数校验。

**使用**：

```python
from prompt_templates import PromptTemplates

messages = PromptTemplates.qa(
    context="检索增强生成（Retrieval-augmented Generation），简称RAG，是当下热门的大模型前沿技术之一。",
    question="RAG 全称是什么？"
)
response = client.chat.completions.create(model="deepseek-v4-flash", messages=messages)
```

## 方法二：Jinja2 模板（适合复杂 Prompt）

Jinja2 模板支持循环、判断、过滤逻辑处理，适合处理复杂条件分支：

```bash
pip install jinja2
```

```python
from jinja2 import Template

# 定义一个复杂的 Prompt 模板
template_str = """
你是一个{{ role }}领域的专家。

{% if style %}
请用{{ style }}的风格回答。
{% endif %}

{% if context %}
## 参考信息
{{ context }}
{% endif %}

{% if examples %}
## 示例
{% for ex in examples %}
输入：{{ ex.input }}
输出：{{ ex.output }}
{% endfor %}
{% endif %}

## 问题
{{ question }}
"""

template = Template(template_str)

prompt_text = template.render(
    role="医疗",
    style="通俗易懂",
    context="头痛可能由多种原因引起，包括紧张性头痛、偏头痛等...",
    examples=[
        {"input": "发烧怎么办", "output": "建议测量体温，如果超过38.5℃..."},
        {"input": "咳嗽吃什么药", "output": "咳嗽需要区分干咳或湿咳..."}
    ],
    question="头痛应该挂什么科？"
)

messages = [
    {"role": "system", "content": "你是一个医疗健康助手"},
    {"role": "user", "content": prompt_text}
]
```

## 方法三：YAML 配置管理（适合多人协作）

对于多人协作场景，需要模板与代码完全分离、多版本管理时选择YAML，非技术人员可编辑。

创建 `prompts.yaml`：

```yaml
templates:
  qa:
    system: "你是一个知识库助手，请基于上下文回答问题。如果无法回答，请如实告知。"
    user: "上下文：\n{context}\n\n问题：\n{question}"
    
  code_review:
    system: "你是一个资深代码审查员，检查以下代码的问题。"
    user: "语言：{language}\n代码：\n```{language}\n{code}\n```"
    
  summary:
    system: "你是一个摘要专家。"
    user: "将以下内容用{max_words}字以内总结：\n\n{text}"
```

```python
import yaml

with open("prompts.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

def apply_template(name: str, **kwargs) -> list:
    tpl = config["templates"][name]
    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user", "content": tpl["user"].format(**kwargs)}
    ]

messages = apply_template("qa", context="...", question="...")
```

## Prompt 工具生态

| 工具 | 用途 | 适用阶段 |
|------|------|----------|
| **LangChain** | Prompt 模板 + Chain + Agent 编排框架 | 生产级应用 |
| **LlamaIndex** | 数据索引 + RAG + Prompt 管理 | 知识库场景 |
| **PromptLayer** | Prompt 版本管理、日志、A/B 测试 | 调试与监控 |
| **PromptBase** | 社区 Prompt 市场，直接购买成熟模板 | 快速参考 |
| **PromptPilot** | Prompt 自动优化工具 | 手动迭代辅助 |

> 注意：先用 Python 原生方式熟练掌握模板化思路，再引入框架。过早引入框架反而会掩盖底层原理。

## 动手实验

1. 创建一个 `PromptManager` 类，支持：
   - 注册模板（按名称）
   - 渲染模板（传入参数）
   - 输出 messages 列表（直接可用于 API 调用）
   - 记录每次调用的 Token 消耗

2. 为以下场景各写一个模板：
   - 产品描述 → 营销文案
   - 代码 Bug 分析
   - 会议纪要提取待办事项
   - 情感分析（正面/负面/中性）

3. 将 P1-P5 的所有练习改写成使用模板的形式

## 完成标准

- [ ] 理解模板化解决的核心问题（复用性、可维护性）
- [ ] 至少用两种方法实现了 Prompt 模板
- [ ] 创建了自己的 `PromptManager` 工具类
- [ ] 重构了之前的代码，统一用模板管理

## 下一步
完成 → 进入 P7-常见陷阱与最佳实践.md
