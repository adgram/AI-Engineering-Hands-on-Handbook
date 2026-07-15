# P4：思维链（Chain-of-Thought, CoT）

## 目标
理解思维链的原理，掌握 Few-shot CoT、自我修正等实用技术，了解 CoT 的进阶扩展

## 核心概念

### 什么是思维链？

让模型在给出最终答案前，先展示一步步的推理过程：

```
普通 Prompt:
问：小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
答：6 个

CoT Prompt:
问：小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
答：开始有 5 个。给小红 2 个：5 - 2 = 3。又买 3 个：3 + 3 = 6。所以答案是 6 个。
```

> 复杂推理需要中间步骤，CoT 让"思考过程"显式化，减少"跳步"导致的逻辑错误。

### Zero-shot CoT

2022 年 Kojima et al. 提出：在 Prompt 末尾加 `"让我们一步步思考"` 可触发模型的逐步推理。**对推理模型几乎无效果**，因为模型内部已在做分步推理。理解此概念有助于把握 CoT 的底层机制。

### Few-shot CoT

使用 `user` + `assistant` 角色的样本形式，在 `assistant` 角色中给出含推理过程的示例，让模型模仿相同的推理风格：

```python
messages = [
    {"role": "system", "content": "你是一个逻辑推理专家，回答前先一步步推理"},
    {"role": "user", "content": "所有猫都怕水。Tom 是一只猫。Tom 怕水吗？"},
    {"role": "assistant", "content": "前提1: 所有猫都怕水。前提2: Tom 是一只猫。根据前提1 和前提2，Tom 是所有猫中的一员，所以 Tom 也怕水。结论: 是的，Tom 怕水。"},
    {"role": "user", "content": "所有会飞的动物都有翅膀。蝙蝠会飞。企鹅是鸟类但不会飞。问题：\n1. 蝙蝠有翅膀吗？\n2. 企鹅是鸟吗？\n3. 所有鸟类都会飞吗？"}
]
```

适用于需要特定推理风格或复杂逻辑的任务。示例中推理步骤越清晰，模型复现越准确。

### 扩展：Generated Knowledge Prompting（外部化 CoT）

与 CoT 让模型在同一轮输出中完成推理不同，Generated Knowledge Prompting 把"生成中间知识"拆成一次独立的 API 调用，再将知识注入第二轮作为上下文回答。把 CoT 的"思考"步骤外部化为开发者可控的多步流程：

```python
# 第一步：生成相关知识
resp_knowledge = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "关于金门大桥的维护，列举几个关键事实"}]
)
knowledge = resp_knowledge.choices[0].message.content

# 第二步：将知识作为上下文，再回答问题
resp_answer = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "利用以下知识回答问题"},
        {"role": "user", "content": f"知识：\n{knowledge}\n\n问题：金门大桥为什么是橙色的？"}
    ]
)
print(resp_answer.choices[0].message.content)
```

> 适用于模型训练数据中不常见或知识截止日期后的信息。先"激活"相关知识再回答，比直接回答准确率更高。

### CoT 的最佳实践（2026 版）

1. **带内置推理的模型**：不需要手动加简单的 CoT 提示词，模型内部已在推理
2. **何时手动加 CoT**：使用非推理模型（开源小模型、旧模型）、需要控制推理格式（Few-shot CoT）、需要结构化输出
3. **警惕过度思考**：CoT 长度与收益呈倒 U 型曲线——简单问题花大量 tokens 推理并不能提高准确率，纯属浪费。**直接回答**、**正常推理**、**过度推理**三者都可能答对，但 token 消耗差数倍到数十倍。简单题用"直接回答"节省成本，复杂题再用 CoT
4. **Generated Knowledge Prompting**：当需要外部知识作答时，考虑将生成知识作为独立步骤

## 动手实验

### 实验 1：Few-shot CoT 实践

运行前文核心概念中的 Few-shot CoT 代码，观察模型如何模仿给出的推理风格。尝试修改 assistant 示例的推理方式来控制风格：

```
示例中推理风格是"前提→推理→结论" → 模型输出相同三段式
示例中推理风格是"直接一步步算" → 模型输出更简洁
```

### 实验 2：过度思考对比

```python
# 用同一道简单题对比三种方式的 token 消耗

question = "2 + 3 × 4 等于多少？"
prompts = {
    "直接回答": "直接回答，不解释：",
    "正常推理": "回答问题，一步步推理：",
    "过度推理": """请非常详细地推理以下问题：
1. 先写出已知条件
2. 逐步计算，每一步都要解释
3. 检查每一步是否有错误
4. 用另一种方法重新验证答案
5. 给出最终答案"""
}
```

### 实验 3：自我修正（Self-Reflection）

先生成初始版本，再让模型审查并优化自己：

```
# 第一步：生成初始版本
写一个 Python 函数，从 JSON 文件中读取配置，然后连接数据库执行指定的 SQL 查询，返回结果。

# 第二步：自我审查
请严格审查以下代码，找出所有问题（错误处理、安全性、资源管理、代码健壮性）：
{initial}
要求：
1. 列出每个问题及严重程度（高/中/低）
2. 每个问题给出原因
3. 提供修复后的完整版本
```

> 自我修正适用于代码审查、文案修改、逻辑验证等场景。模型在"审查者"角色下会比"创作者"角色更严格地发现问题。

### 实验 4：Self-Consistency（自洽性）

多次运行后通过关键词统计找出模型最倾向的表达，适用于开放生成任务：

```python
import re
from collections import Counter

topic = "人工智能对教育的影响"
answers = []
for i in range(5):
    resp = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": f"第{i+1}轮：用一句话概括{topic}。"}],
    )
    answers.append(resp.choices[0].message.content.strip())

all_words = []
for a in answers:
    words = re.findall(r'[\u4e00-\u9fff]{2,}', a)
    all_words.extend(words[:3])

word_freq = Counter(all_words).most_common(5)
for w, c in word_freq:
    print(f"「{w}」: {c}次")
```

> 推理模型输出已足够稳定，实际生产中 Self-Consistency 的使用频率较低。了解其原理即可。

### 实验 5：层次推理

先规划再执行——让模型先列出所需推理步骤，再逐条执行：

```
# 直接推理
回答问题：

# 层次推理：先规划再执行
先列出解决这个问题需要的所有推理步骤，然后逐一执行每个步骤，最后给出答案。
问题：
```

> 适合约束条件多、步骤依赖复杂的任务。层次推理的效果在于输出结构更清晰，有明确的分步结构，可读性和可审计性更高。

### 实验 6：XML 标签预填充

用自定义标签规范 CoT 的输出结构，便于下游解析：

```python
prompt = """
<question>
小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
</question>

<reasoning>
逐步推理：
</reasoning>
<answer>
</answer>
"""
```

> 适合需要对推理和答案做单独处理的场景（如只提取最终答案做下游流程）。推理模型输出本身已有 `reasoning_content` 和 `content`，预填充标签的实际需求较少。
>
> 详细格式化输出，可以看下一章 [P5-结构化输出与Function-Calling基础](P5-结构化输出与Function-Calling.md)

## 完成标准

- [ ] 理解 CoT 为什么能提升推理能力
- [ ] 实现了一个 Few-shot CoT 示例
- [ ] 理解 Token 收益递减，知道何时用"直接回答"节约成本
- [ ] 理解 Generated Knowledge Prompting 与 CoT 的关系
- [ ] 能判断什么场景该用 CoT，什么场景不该用

## 下一步

完成 → 进入 [P5-结构化输出与Function-Calling基础](P5-结构化输出与Function-Calling.md)

