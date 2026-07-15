# P3：Prompt 自动优化与循环工程（Loop Engineering）

## 目标
从手动迭代优化升级到系统化的自动优化体系，掌握 Loop Engineering、GEPA、Auto Research 等前沿方法论。

> **定位**：Day1 P7 定义了 Prompt 评估的 5 维度框架（`evaluate_prompt`），本篇将其升级为自动化的优化循环。本篇的基于错误驱动的优化思想是 Agent 阶段（错误处理与重试、Agent 评估）自动恢复体系的基础。

---

## 第一部分：手动迭代优化

### 迭代优化案例：餐厅菜单改版

一个真实案例展示 Prompt 如何从"能用"迭代到"好用"：

**V1 — 基础版**：
```
将以下菜单翻译成英文：
[菜单内容]
```
→ 问题：翻译不完整，格式混乱

**V2 — 加格式要求**：
```
将以下菜单翻译成英文，保持 Markdown 表格格式：
[菜单内容]
```
→ 问题：菜名翻译不统一（"麻婆豆腐"有时 Mapo Tofu 有时 Tofu Mapo）

**V3 — 加术语表**：
```
将以下菜单翻译成英文，保持 Markdown 表格格式。
术语对照：麻婆豆腐=Mapo Tofu，宫保鸡丁=Kung Pao Chicken
[菜单内容]
```
→ 问题：价格格式不一致（¥38 vs 38 yuan）

**V4 — 完整约束**：
```
将以下菜单翻译成英文：
- 保持 Markdown 表格
- 菜名使用官方翻译（术语表见下方）
- 价格统一格式：¥{数字}
- 翻译后菜名首字母大写
- 不要翻译餐厅名称

术语表：麻婆豆腐=Mapo Tofu，宫保鸡丁=Kung Pao Chicken
[菜单内容]
```
→ 结果：格式统一、翻译准确、可直接上线

> 每次优化只解决一个错误模式，迭代 3-5 轮后再统一测试，避免一次改太多无法定位问题。

### 基于指标的迭代流程

```
原始 Prompt
    ↓
定义评估指标（准确率、格式合规率、用户评分）
    ↓
生成测试集（10~20 个测试用例）
    ↓
跑基线结果
    ↓
分析错误模式 → 针对性修改 Prompt
    ↓
重新测试 → 对比新旧结果
    ↓
重复直到满意
```

### 评估驱动的优化

下面演示了三种提示词对判断结果的影响：

```python
test_cases = [
    {"input": "产品：某品牌手机，用户说'信号很好，但有点重'", "expected": "正面"},
    {"input": "产品：某品牌蓝牙耳机，用户说'用了三天就坏了'", "expected": "负面"},
    {"input": "产品：某咖啡，用户说'味道一般，价格适中'", "expected": "中性"},
]

# 注：Day1 P7 提供了更全面的 5 维度评估框架（准确率/完整性/一致性/安全性/体验），此处简化

v1 = "分析用户评论的情感倾向，只输出正面、负面或中性"
print("=== Prompt V1 ===")
acc1 = evaluate_accuracy(v1, test_cases)

v2 = "你是一个情感分析专家。分析以下评论的情感倾向。\n规则：\n1. 只输出正面、负面或中性\n2. 如果评论中既有正面也有负面，以整体倾向为准\n3. 不要输出任何解释\n\n"
print("\n=== Prompt V2 ===")
acc2 = evaluate_accuracy(v2, test_cases)

v3 = v2 + "\n\n示例：\n评论：'性价比很高，推荐' → 正面\n评论：'质量太差了' → 负面\n评论：'还行吧，能用' → 中性"
print("\n=== Prompt V3 ===")
acc3 = evaluate_accuracy(v3, test_cases)

print(f"\n=== 总结 ===")
print(f"V1: {acc1:.0%} → V2: {acc2:.0%} → V3: {acc3:.0%}")
```

> 手动迭代的关键问题是**每轮都靠人盯着**——人写 Prompt、人看结果、人分析错误、人改。能不能让 AI 自己完成这个闭环？下面介绍三种自动优化范式。

---

## 第二部分：自动优化的三种范式

三种范式共享同一个核心思想——**反思式进化**：让 AI 从自己的错误中学习。它们都围绕三个角色展开：

| 角色 | 职责 | 类比 |
|------|------|------|
| **Generator** | 基于当前 Prompt + 失败案例，生成候选变体 | 提出方案的设计师 |
| **Evaluator** | 用测试集评估候选 Prompt，输出评分 + 错误分析 | 质检员 |
| **Optimizer** | 分析失败模式，给出改进方向，驱动下一轮 | 复盘导师 |

区别在于实现方式：GEPA 用多角色分工、Loop Engineering 强调工程化落地、SPO 用单一模型自我反思。

### 范式一：GEPA（Generative Evolutionary Prompt Adapter）

GEPA 的核心思想是**反思式进化**——让 AI 从自己的错误中学习，而不是被动等待人工调参。

#### 工作流程

```
Generator 生成 N 个候选 Prompt
        ↓
Evaluator 对每个候选跑测试集评分
        ↓
选出分数最高的候选
        ↓
Optimizer 分析失败案例，提取模式
        ↓
把改进建议反馈给 Generator → 下一轮
```

#### 简化 GEPA 实现

```python
class GEPAEngine:
    def __init__(self, llm_client, test_cases, evaluator_fn):
        self.llm = llm_client
        self.test_cases = test_cases
        self.evaluate = evaluator_fn
        self.history = []

    def generate_candidates(self, current_prompt, error_analysis, n=3):
        """Generator：基于当前 Prompt + 错误分析生成变体"""
        candidates = []
        for i in range(n):
            resp = self.llm.chat(
                messages=[{"role": "system", "content": "你是一个 Prompt 优化专家。"},
                          {"role": "user", "content": f"当前 Prompt：\n{current_prompt}\n\n 上轮错误分析：\n{error_analysis}\n\n 请生成一个改进版本（变体 {i+1}）。只返回 Prompt 本身。"}]
            )
            candidates.append(resp.choices[0].message.content)
        return candidates

    def analyze_errors(self, results):
        """Optimizer：分析失败案例，提取模式"""
        failures = [r for r in results if not r["correct"]]
        if not failures:
            return "没有发现错误，当前 Prompt 表现良好。"
        error_text = "\n".join([f"输入: {f['input']}\n期望: {f['expected']}\n得到: {f['output']}" for f in failures[:5]])
        resp = self.llm.chat(
            messages=[{"role": "system", "content": "你是一个 Prompt 错误分析专家。"},
                      {"role": "user", "content": f"分析以下失败案例的错误模式，给出改进方向的建议：\n{error_text}"}]
        )
        return resp.choices[0].message.content

    def run_epoch(self, current_prompt):
        results = self.evaluate(current_prompt, self.test_cases)
        error_analysis = self.analyze_errors(results)
        candidates = self.generate_candidates(current_prompt, error_analysis)
        best_score, best_prompt = 0, current_prompt
        for cand in candidates:
            score = self.evaluate(cand, self.test_cases)
            if score > best_score:
                best_score, best_prompt = score, cand
        self.history.append({"prompt": best_prompt, "score": best_score, "errors": error_analysis})
        return best_prompt, best_score
```

> GEPA 用多角色分工实现了自动优化，但生产环境还需要考虑更多工程问题——如何防止作弊、如何确保不倒退、如何精确定位问题。Loop Engineering 正是在 GEPA 基础之上解决这些问题的工程化方案。

### 范式二：Loop Engineering（循环工程）

#### Concept：从"写 Prompt"到"设计循环"

传统方式是把 Prompt 当作一次性文案来写；Loop Engineering 把 Prompt 优化看作**一个可编程的闭环系统**：

```
Prompt → 执行 → 评估 → 分析 → 改写 → Prompt'
  ↑                                  |
  └──────────────────────────────────┘
```

核心信条：**不要手动调 Prompt，让循环自动调。**

#### 工程化角色扩展

与 GEPA 相同的三角结构，但 Loop Engineering 在工程落地层面对各角色增加了具体的输入/输出/约束约定（对比公共表，此处是工程化细化）：

| 角色 | 输入 | 输出 | 关键约束 |
|------|------|------|---------|
| Generator | 当前 Prompt + 错误日志 | N 个候选 Prompt | 每次只改 1-2 个 section |
| Evaluator | 候选 Prompt + 测试集 | 评分 + 逐条结果 | 使用独立评分 Prompt |
| Optimizer | 失败案例 | 错误模式 + 改进方向 | 只分析不修改 |

#### 实战案例：通过迭代提高准确率

场景：让模型从医疗文本中提取药物名称。

```
基线 Prompt（V0）:
"从以下文本中提取所有药物名称。"

准确率：14%（只识别了最常见的阿莫西林，漏掉了商品名和复方制剂）

↓ 第一轮 Loop

Generator 分析失败案例后发现：
  - 模型不认识商品名（"泰诺"、"白加黑"）
  - 复方制剂只提取了一半（"氨酚黄那敏"只提取了"氨酚"）
  - 剂量信息被当作药物名提取

改进后的 Prompt（V1）:
"从以下文本中提取所有药物名称。
规则：
1. 包括化学名（阿莫西林）、商品名（泰诺）、复方制剂（氨酚黄那敏颗粒）
2. 排除：剂量（500mg）、剂型（片剂、胶囊）、给药途径
3. 同一药物的商品名和化学名只保留一个"

准确率：67%

↓ 第二轮 Loop

Optimizer 发现新错误模式：
  - 中药方剂被遗漏（"黄芪 15g，当归 10g"→ 应提取药材名）
  - 疫苗名称（"流感疫苗"）被排除

改进后的 Prompt（V2）:
"从以下文本中提取所有药物名称。
规则：
1-3 同 V1
4. 中药方剂中的每味药材单独提取
5. 疫苗、生物制剂也算药物
6. 输出格式：每行一个药物名，不要编号"

准确率：98%
```

> 关键洞察：**不是 Prompt 越长越好，而是错误覆盖越全越好。** 每次循环只解决 1-2 种错误模式。

#### 简化 Loop Engine 实现

```python
import json
from dataclasses import dataclass, field
from typing import List, Callable

@dataclass
class LoopConfig:
    prompt_sections: dict  # {"指令": "...", "规则": "...", "示例": "..."}
    test_cases: List[dict]
    evaluator_fn: Callable
    llm_client: any
    max_rounds: int = 5
    history: list = field(default_factory=list)

class LoopEngine:
    def __init__(self, config: LoopConfig):
        self.cfg = config
        self.best_prompt = dict(config.prompt_sections)
        self.best_score = 0.0

    def assemble_prompt(self, sections: dict) -> str:
        return "\n\n".join([f"### {k}\n{v}" for k, v in sections.items()])

    def generate_variant(self, sections: dict, error_report: str) -> dict:
        """Generator：增量修改，每轮只改 1-2 个 section"""
        prompt_text = self.assemble_prompt(sections)
        resp = self.cfg.llm_client.chat(
            messages=[{"role": "system", "content": "你是一个 Prompt 优化专家。每次只修改 1-2 个 section。"},
                      {"role": "user", "content": f"当前 Prompt sections：\n{json.dumps(sections, ensure_ascii=False, indent=2)}\n\n 错误分析：\n{error_report}\n\n 请返回修改后的完整 sections JSON（保持其他 section 不变）。"}]
        )
        return json.loads(resp.choices[0].message.content)

    def optimize(self):
        for round_idx in range(self.cfg.max_rounds):
            prompt_text = self.assemble_prompt(self.best_prompt)
            score = self.cfg.evaluator_fn(prompt_text, self.cfg.test_cases)
            failures = self._collect_failures(prompt_text)

            # Optimizer 分析
            error_report = self._analyze(failures) if failures else ""

            # Generator 生成变体
            if failures:
                new_sections = self.generate_variant(self.best_prompt, error_report)
                new_score = self.cfg.evaluator_fn(self.assemble_prompt(new_sections), self.cfg.test_cases)
                # 棘轮机制：只保留分数不降的版本
                if new_score >= score:
                    self.best_prompt = new_sections
                    self.best_score = new_score

            self.cfg.history.append({"round": round_idx, "score": score, "failures": len(failures)})
            print(f"Round {round_idx+1}: score={score:.0%}, failures={len(failures)}")
            if not failures:
                break

    def _collect_failures(self, prompt_text) -> List[dict]:
        results = []
        for case in self.cfg.test_cases:
            resp = self.cfg.llm_client.chat(
                messages=[{"role": "system", "content": prompt_text},
                          {"role": "user", "content": case["input"]}]
            )
            output = resp.choices[0].message.content
            if case["expected"] not in output:
                results.append({"input": case["input"], "expected": case["expected"], "output": output})
        return results

    def _analyze(self, failures: List[dict]) -> str:
        text = "\n".join([json.dumps(f, ensure_ascii=False) for f in failures[:5]])
        resp = self.cfg.llm_client.chat(
            messages=[{"role": "system", "content": "分析以下失败案例的错误模式，用中文总结。"},
                      {"role": "user", "content": text}]
        )
        return resp.choices[0].message.content
```

> GEPA 和 Loop Engineering 都依赖多角色分工来避免偏见。还有一种更轻量的思路——让同一个模型自己反思自己的输出，这就是 SPO（Self-Reflective Prompt Optimization）。

### 范式三：SPO（自我反思）

#### Self-Reflective Prompt Optimization

SPO 的核心思想是让大模型通过**自我反思**来优化提示词，形成"分析 → 批判 → 重写 → 评估"的闭环。

与 GEPA 和 Loop Engineering 的多角色分工不同，SPO 只用**一个模型、一个角色**完成全部流程。代价是自评虚高风险更大（见上文自评虚高陷阱），因此必须引入棘轮机制。

| 维度 | SPO | GEPA | Loop Engineering |
|------|-----|------|-----------------|
| 角色数 | 1（自问自答） | 3（生成/评估/分析分离） | 3（带工程约束） |
| 实现成本 | 最低（50 行） | 中等（100 行） | 较高（150+ 行） |
| 自评虚高 | 最严重 | 较低 | 最低（三重隔离） |
| 适用场景 | 快速原型、个人实验 | 中小团队常规项目 | 生产级、需要回滚审计 |
| 收敛速度 | 快（单轮就能见效） | 中等 | 慢（但最稳定） |

> **选型建议**：SPO 适合 1-2 轮快速验证，一旦超过 3 轮仍有波动，应切换为 GEPA 或 Loop Engineering 来获得更稳定的收敛。

#### 工作流程

```
分析当前 Prompt 的输出结果
        ↓
批判：指出问题所在（"输出格式不一致"、"边界情况没覆盖"）
        ↓
重写：基于批判生成新版本
        ↓
评估：对比新旧版本的性能
        ↓
循环直到收敛
```

#### 简化 SPO 实现

```python
def spo_round(current_prompt, test_cases, llm_client):
    """一次 SPO 迭代"""
    # Step 1: 运行当前 Prompt
    results = run_prompt(current_prompt, test_cases, llm_client)
    failures = [r for r in results if not r["correct"]]

    # Step 2: 自我批判
    critique_resp = llm_client.chat(
        messages=[{"role": "system", "content": "你是一个严格的 Prompt 评审专家。分析以下 Prompt 和失败案例，指出 3 个具体问题。"},
                  {"role": "user", "content": f"Prompt：\n{current_prompt}\n\n失败案例：\n" +
                   "\n".join([f"输入: {f['input']}\n输出: {f['output']}\n期望: {f['expected']}" for f in failures[:5]])}]
    )
    critique = critique_resp.choices[0].message.content

    # Step 3: 基于批判重写
    rewrite_resp = llm_client.chat(
        messages=[{"role": "system", "content": "基于评审意见重写以下 Prompt。保持原意，修复指出的问题。"},
                  {"role": "user", "content": f"原 Prompt：\n{current_prompt}\n\n评审意见：\n{critique}\n\n重写后的 Prompt："}]
    )
    new_prompt = rewrite_resp.choices[0].message.content

    # Step 4: 评估新版本
    new_results = run_prompt(new_prompt, test_cases, llm_client)
    new_correct = sum(1 for r in new_results if r["correct"])
    old_correct = sum(1 for r in results if r["correct"])
    print(f"旧版: {old_correct}/{len(test_cases)} → 新版: {new_correct}/{len(test_cases)}")

    return new_prompt if new_correct >= old_correct else current_prompt
```

> 三种范式对比：**GEPA** 多角色分工最鲁棒，**Loop Engineering** 工程化最完整，**SPO** 自我反思最轻量。选择哪种取决于具体场景——需要稳定性选前两者，快速原型选 SPO。

> 无论用哪种范式，实战中都积累了一些通用经验和陷阱。下面这些发现来自大规模 Prompt 优化的真实案例。

## 第三部分：通用工程规范与实战经验

以下原则分为两类：优化流程的执行约束，以及大规模实验中提炼的策略洞见。

### 优化流程执行约束

#### 增量修改策略

把 Prompt 拆成多个 section，每轮只改 1-2 个：

```json
{
  "角色设定": "你是一个情感分析专家。",
  "指令": "分析以下评论的情感倾向。",
  "规则": [
    "只输出正面、负面或中性",
    "以整体倾向为准",
    "不要输出任何解释"
  ],
  "示例": [
    "评论：'性价比很高' → 正面",
    "评论：'质量太差了' → 负面"
  ],
  "输出格式": "仅输出一个词"
}
```

> 拆分后可以精确定位问题：格式错误改"输出格式"，边界模糊改"规则"，不需要每次都重写全部。

#### 三重隔离架构

避免优化过程中的"作弊"行为，必须隔离三个环境：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   生成环境       │     │   评分环境        │     │   迭代 Agent    │
│                 │     │                 │     │                 │
│ 能看到：         │     │ 能看到：          │     │ 能看到：         │
│ · Prompt 原文   │     │ · 评分标准        │     │ · 评分结果       │
│ · 用户输入       │     │ · 模型输出        │     │ · 错误案例       │
│                 │     │ · 参考答案       │     │                 │
│ 不能看到：       │      │ 不能看到：        │     │ 不能看到：        │
│ · 评分标准       │     │ · Prompt 原文    │     │ · 原始输入       │
│ · 参考答案       │     │ · 优化策略        │     │ （防过拟合）      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### 自评虚高陷阱

**核心规则：修改 Prompt 的 AI 不能给自己打分。**

如果同一个 AI 既修改 Prompt 又评估结果，会产生严重的**自评虚高**——AI 会潜意识地按自己 Prompt 的"意图"评分，而非真实质量。

```python
# f：同一个模型既改又评
improver = "deepseek-v4-flash"
evaluator = "deepseek-v4-flash"  # 分数会虚高 10-20%

# 正例：评估用不同模型或确定性规则
improver = "deepseek-v4-flash"
evaluator = "gpt-4o"  # 或使用完全基于规则的评估
```

#### 棘轮机制（Ratchet）

每次迭代后，**只保留分数不降的版本**，拒绝任何回退：

```
Round 1:  71%  → 保留
Round 2:  85%  → 保留（提升）
Round 3:  82%  → 丢弃（回退到 R2 的 85%）
Round 4:  91%  → 保留（新最佳）
Round 5:  89%  → 丢弃（回退到 R4 的 91%）
```

```python
class Ratchet:
    def __init__(self):
        self.best_prompt = None
        self.best_score = -1.0
        self.log = []

    def try_update(self, prompt, score):
        if score >= self.best_score:
            self.best_prompt = prompt
            self.best_score = score
            self.log.append({"action": "KEEP", "score": score})
            return True
        else:
            self.log.append({"action": "DISCARD", "score": score, "best_so_far": self.best_score})
            return False
```

### 数据与复盘规范

#### 失败记录不删除

每次迭代的失败案例全部记录在实验日志中，用于后续的全局分析：

```python
experiment_log = {
    "metadata": {"task": "情感分析", "model": "deepseek-v4-flash", "test_size": 100},
    "rounds": [
        {"round": 1, "prompt": "...", "score": 0.71, "failures": [{"input": "...", "output": "...", "expected": "..."}]},
        {"round": 2, "prompt": "...", "score": 0.85, "failures": [...]},
    ]
}
# 失败数据可汇总分析长期趋势
```

#### 四个关键发现

1. Prompt 不是越长越好

一个真实案例：7.8KB 的复杂 Prompt（包含大量示例和规则）优化到 2.7KB 后，准确率反而提升了。

> 原因：过长 Prompt 分散模型注意力，关键指令被淹没。精简到核心约束后，模型更聚焦。

2. 覆盖率和简洁性天然矛盾

```
覆盖所有边界情况 → Prompt 膨胀 → 模型注意力稀释 → 核心能力下降
        ↑                                     ↓
         ← ← ← 用信息密度打破循环 ← ← ← ← ← ← ← ←
```

**信息密度 = 每千 token 解决的问题数**。优化方向不是"加更多规则"，而是"用更少的词覆盖更多场景"。

3. 小样本高分不代表真的好

| 测试集大小 | Prompt A 得分 | Prompt B 得分 | 结论 |
|-----------|-------------|-------------|------|
| 10 条 | 100% | 90% | A 更好？ |
| 100 条 | 82% | 91% | B 实际更好 |

> 小样本容易过拟合。最少用 50 条测试集，100+ 条才可靠。

4. 连续失败应放弃

如果在某个方向上连续 3 轮没有提升，果断换方向：

```
# 反例：在"加示例"上硬撑 10 轮
Round 4: 加了 2 个反例 → 85% → 没提升
Round 5: 加了 3 个反例 → 84% → 回退
Round 6: 加了 5 个反例 → 85% → 原地踏步

# 正例：换方向
Round 4: (加示例) 85% → 没提升
Round 5: (改规则结构) 91% → 有效！继续这个方向
```

> 规则：连续 3 轮提升 < 1% → 立即切换优化策略。

---

## 第四部分：DSPy 编译器范式

前三个部分都围绕"如何更好地写 Prompt"展开。DSPy 的作用则是 不写 Prompt，只需声明输入/输出的 Schema，编译器自动生成并优化 Prompt。

### DSPy 基础概念

```bash
pip install dspy
```

```python
import dspy

lm = dspy.LM("openai/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com")
dspy.configure(lm=lm)

class SentimentAnalysis(dspy.Signature):
    """分析评论的情感倾向。"""
    review = dspy.InputField()
    sentiment = dspy.OutputField(desc="正面、负面或中性")

class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(SentimentAnalysis)

    def forward(self, review):
        return self.classify(review=review)

def validate_sentiment(example, pred, trace=None):
    return example.sentiment == pred.sentiment

classifier = SentimentClassifier()
```

> 1. **Signature（签名）**：声明 `review → sentiment` 的输入输出字段，代替手写"分析评论情感"的 Prompt
> 2. **Module（模块）**：`SentimentClassifier` 封装了 `ChainOfThought(SentimentAnalysis)`，模型自动按"思考→推理→输出"的方式处理
> 3. **验证函数**：`validate_sentiment` 定义"什么算正确"，框架据此自优化
> 4. **全程无字符串**：没有任何一行 Prompt 文本，全是 Python 类型声明

DSPy 后续版本（3.0+）进一步强化了声明式能力：

### DSPy 3.0 的声明式 Schema

DSPy 3.0 引入的核心概念：**不再手写 Prompt，而是声明输入/输出的 Schema，编译器自动生成优化后的 Prompt。**

```
传统方式：
  手写 Prompt → 测试 → 修改 → 测试...

DSPy 3.0 方式：
  定义 Signature → 编译器自动生成 → 评估 → 编译器自动优化
```

### 手写 vs 声明式对比

| 方式 | 代码量 | 可维护性 | 优化能力 | 迁移成本 |
|------|--------|---------|---------|---------|
| 手写 Prompt | 需反复调参 | 低（文案耦合） | 靠人工经验 | 换模型需重写 |
| 声明式 Signature | 定义一次 | 高（与模型解耦） | 编译器自动调优 | 换模型只需换 LM 配置 |

### 代码示例对比

```python
# ====== 传统手写方式 ======
from openai import OpenAI
client = OpenAI(api_key="sk-your-key", base_url="https://api.deepseek.com")

patient_text = "患者：李某，男，45岁。主诉：咳嗽两周，痰黄。用药：阿莫西林胶囊 0.5g tid，复方甘草口服液 10ml tid。"

prompt = """从以下患者记录中提取药物名称。
规则：
- 只提取药物名称，不提取剂量、频次
- 商品名和化学名都算
- 中药方剂每味药材单独提取

患者记录：{text}
药物："""

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt.format(text=patient_text)}]
)
handwritten_result = response.choices[0].message.content
print(f"[手写方式] 提取结果: {handwritten_result}")

# ====== DSPy 3.0 声明式方式 ======
import dspy
from typing import List

lm = dspy.LM("openai/deepseek-chat", api_key="sk-your-key", api_base="https://api.deepseek.com")
dspy.configure(lm=lm)

class DrugExtraction(dspy.Signature):
    """从患者记录中提取所有药物名称（包括化学名、商品名、中药药材、疫苗）。"""
    patient_note: str = dspy.InputField(desc="患者记录文本")
    drugs: List[str] = dspy.OutputField(desc="提取的药物名称列表")

class DrugExtractor(dspy.Module):
    def __init__(self):
        self.extractor = dspy.Predict(DrugExtraction)

    def forward(self, patient_note):
        return self.extractor(patient_note=patient_note)

# 准备训练数据 & 验证函数
trainset = [
    dspy.Example(patient_note="患者服用头孢克肟和布洛芬", drugs=["头孢克肟", "布洛芬"]).with_inputs("patient_note"),
    dspy.Example(patient_note="处方：阿奇霉素、氯雷他定", drugs=["阿奇霉素", "氯雷他定"]).with_inputs("patient_note"),
]

def validate_drugs(example, pred, trace=None):
    return set(example.drugs) == set(pred.drugs)

# 编译器自动优化
extractor = DrugExtractor()
optimizer = dspy.teleprompt.BootstrapFewShot(metric=validate_drugs)
optimized = optimizer.compile(extractor, trainset=trainset)
```

> 声明式的核心优势：**换模型只需改一行 `lm = dspy.LM(...)`**，编译器会自动适配不同模型的 Prompt 风格。

### DSPy 内置优化器

| 优化器 | 策略 | 适用场景 |
|--------|------|---------|
| `BootstrapFewShot` | 从训练集构建少样本示例 | 有 10-100 条标注数据 |
| `BootstrapFewShotWithRandomSearch` | 在示例组合中随机搜索 | 需要更强的泛化性 |
| `MIPROv2` | 贝叶斯搜索指令+示例联合优化 | 大规模自动化调优（对应前文范式一 GEPA） |
| `COPRO` | 指令级坐标上升 | Prompt 文本结构的逐轮优化 |

对比前文的 GEPA 范式，MIPROv2 代表了另一种自动优化哲学：

| 维度 | MIPROv2 | GEPA |
|------|---------|------|
| 优化策略 | 贝叶斯搜索 + 指令/示例联合优化 | 进化式生成 + 反思反馈 |
| 透明度 | 黑盒优化，看不到中间 Prompt | 白盒，每轮 Prompt 可审查 |
| 失败利用 | 隐式（靠采样避开） | 显式（分析错误并指导改进） |
| 适用场景 | 大规模自动化调优 | 需要可解释性的小到中型任务 |
| 实现复杂度 | 高（依赖 DSPy 编译器） | 低（100 行即可实现原型） |

```python
# 选择优化器的通用模式
optimizer = dspy.teleprompt.BootstrapFewShot(metric=validate_fn)
optimized_program = optimizer.compile(program, trainset=trainset)
```

---

## 第五部分：启发式规则速查表

除了自动优化，下面是一些启发式方法来提升 Prompt 质量：

| 症状 | 根因 | 优化方向 | 示例 |
|------|------|---------|------|
| 输出格式不稳定 | 模型不知道期望的结构 | 给具体输出模板 | "输出格式：{'sentiment': '正面'}" |
| 回答过于啰嗦 | 没有长度锚点 | 加长度约束 | "不超过 20 个字" |
| 回答过于简略 | 没有复杂度锚点 | 要求分点阐述 | "请用三个要点回答" |
| 忽略关键约束 | 核心指令淹没在细节中 | 调整指令位置 | 关键指令放在开头和结尾 |
| 回答不准确/幻觉 | 采样温度过高 | 增加确定性 | temperature=0 |
| 边界情况处理差 | 模型不知道不希望出现什么 | 加反例（负样本） | "不要输出'无法判断'" |
| 术语翻译不一致 | 模型没有标准参考 | 加术语表 | "麻婆豆腐=Mapo Tofu" |
| 多步推理出错 | 模型跳过中间步骤 | 要求显式推理 | "请先写出分析过程，再给出结论" |
| 回答风格不统一 | 没有风格锚点 | 加角色设定 | "你是一个严谨的医学编辑" |

> 这些规则的核心思路都是**给模型提供锚点**——要么是格式锚点、长度锚点、术语锚点，要么是过程锚点。锚点越具体，输出越可控。

---

## 综合实验：从手动到自动的完整流程

下面结合本篇所有概念——手动基线、评估驱动、GEPA/Loop Engineering 的多角色结构、Auto Research 的失败记录——实现一个完整的自动优化管道：

```python
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent.parent/"common"))
from llm_client import LLMClient
client = LLMClient()

# 1. 定义任务和测试集
test_cases = [
    {"input": "这款手机拍照一流，但续航一般", "expected": "正面"},
    {"input": "用了三天就坏了，质量太差", "expected": "负面"},
    {"input": "价格适中，功能齐全", "expected": "正面"},
    {"input": "一般般吧，没什么特别的", "expected": "中性"},
    {"input": "客服态度差还不给退款", "expected": "负面"},
    {"input": "包装精美，送人很有面子", "expected": "正面"},
    {"input": "功能很多但大部分用不上", "expected": "中性"},
    {"input": "性价比很高，推荐购买", "expected": "正面"},
    {"input": "物流太慢了等了一个星期", "expected": "负面"},
    {"input": "和描述一致，没有惊喜也没有失望", "expected": "中性"},
]

# 2. 定义评估函数
def eval_fn(prompt, cases):
    correct = 0
    for c in cases:
        resp = client.chat(
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": c["input"]}]
        )
        if c["expected"] in resp.choices[0].message.content:
            correct += 1
    return correct / len(cases)

# 3. 手动基线
base_prompt = "分析评论的情感：正面、负面或中性"
base_score = eval_fn(base_prompt, test_cases)
print(f"基线准确率: {base_score:.0%}")

# 4. 运行 Loop Engine
config = LoopConfig(
    prompt_sections={
        "角色设定": "你是一个情感分析专家。",
        "指令": "分析以下评论的情感倾向。",
        "规则": "只输出正面、负面或中性",
        "输出格式": "仅输出一个词"
    },
    test_cases=test_cases,
    evaluator_fn=eval_fn,
    llm_client=client,
    max_rounds=3
)
engine = LoopEngine(config)
engine.optimize()

# 5. 查看优化历史
for h in engine.cfg.history:
    print(f"轮次 {h['round']+1}: 准确率={h['score']:.0%}, 失败数={h['failures']}")
```

### 实验提示

- **实验 1**：运行手动迭代（V1→V2→V3），记录每次准确率变化
- **实验 2**：运行 Loop Engine，对比自动优化与手动迭代的收敛速度
- **实验 3**：修改 `max_rounds` 观察更多轮次后的边际收益递减
- **实验 4**：尝试 SPO 单轮自优化，看能否达到 Loop Engine 多轮效果

### 优化成本考量

虽然细致的优化可以大幅降低每次调用的 token 消耗，但需要注意：
- **按调用次数计费的套餐**：每轮自动优化需要多次 LLM 调用（评估→分析→生成变体→再评估），迭代多轮后总调用次数激增，对按次计费的套餐并不友好
- **思考深度 vs 迭代次数**：多次低 `reasoning_effort` 调用不一定比一次高 `reasoning_effort` 调用更快——如果单次深思考能直接定位根因，反复浅层优化反而更耗时
- **建议**：先用少量测试集（5-10 条）做粗调，收敛后再用完整测试集精调，避免在早期浪费调用次数

---

## 完成标准

- [ ] 理解了从手动迭代到自动优化的演进路径
- [ ] 掌握了 Loop Engineering 的 Generator→Evaluator→Optimizer 三角结构
- [ ] 理解了三重隔离架构和棘轮机制的原理
- [ ] 能够用 100 行左右实现一个简化版自动优化引擎
- [ ] 了解了 DSPy 3.0 声明式 Schema 的核心理念
- [ ] 知道何时该换方向而非硬撑

> Loop Engineering 的 Generator-Evaluator-Optimizer 三角结构将在 Agent 阶段的 Tool Call 循环（Day7-P3）和 Agent 评估（Day9-P6）中再次出现，届时自动恢复策略将复用本节的自优化思想。

---

## 下一步 → [P4-对抗性Prompt与安全](P4-对抗性Prompt与安全.md)
