# P7：小项目 — 带评估的 RAG Pipeline

## 目标
整合本周所有知识，构建一个完整、可评估的 RAG Pipeline。

|          | 学习路径                                                     |
| -------- | ------------------------------------------------------------ |
| 已有基础 | Day4-P1~P6 全部 + Day3-P7（模块化项目结构） |
| 本章内容 | 整合检索策略×Rerank×k 值，通过网格搜索找到最优配置组合，形成"策略选择→评估→优化"的实验闭环。 |

## 项目结构

项目采用模块化设计，每个文件职责单一，便于独立开发和替换各组件。

```
rag_pipeline/            # learn_prompt/rag_pipeline/
├── __init__.py          # 包标识
├── main.py              # 入口（CLI）
├── dataset.py           # 测试集管理
├── config.py            # 配置
└── run_experiment.py    # 实验管理器（网格搜索）
```

## 核心实现

Pipeline 由各核心模块组成，每个模块职责单一，可灵活组合。

### retriever.py — 多策略检索

支持向量检索、HyDE 检索和多路召回三种策略，通过统一接口调用。

> `vector_search` / `hyde_search` / `multi_route_search` / `retrieve` 实现详见 Day4-P5（高级检索 HyDE 与多路召回）

### reranker.py — API 重排序

通过 SiliconFlow API 调用 BGE-Reranker 模型，对检索结果精排。

> `api_rerank` 实现详见 Day4-P6（Rerank 重排序）

### generator.py — 生成器

将检索结果组装成 Prompt，调用 LLM 生成回答。支持 standard 和 citation 两种策略。

> Prompt 组装策略（standard / citation）原理详见 Day4-P2（Prompt 中注入上下文的最佳实践）

### evaluator.py — 评估器

使用 LLM-as-Judge 评估回答的忠实度和相关性，输出结构化 JSON。

> `evaluate_faithfulness` / `evaluate_relevance` 详见 Day4-P3（检索质量评估）和 Day4-P4（生成质量评估）

### dataset.py — 测试集管理

定义测试用例，覆盖 RAG 核心知识点。

```python
TEST_CASES = [
    {"question": "RAG 是什么？它的工作流程是怎样的？"},
    {"question": "Rerank 重排序模型有什么作用？"},
    {"question": "HyDE 方法如何改善检索效果？"},
    {"question": "如何评估 RAG 系统的质量？"},
]

def get_test_cases() -> list:
    return TEST_CASES
```

### run_experiment.py — 实验管理器

ExperimentRunner 是实验调度核心，支持多策略检索、可选 Rerank、自动评估和网格搜索。

核心逻辑（导入的 `retrieve` / `api_rerank` / `Generator` / `evaluate_faithfulness` / `evaluate_relevance` 分别详见 Day4-P5 / P6 / P2 / P3-P4）：

```python
class ExperimentRunner:
    def __init__(self, store, llm, model: str = "deepseek-v4-flash"):
        self.store = store
        self.llm = llm
        self.model = model
        self.generator = Generator(llm, model)

    def run_config(self, test_cases: list, config: dict) -> dict:
        for case in test_cases:
            docs = retrieve(config["retriever"], self.store, self.llm, case["question"], config.get("k", 5))
            if config.get("rerank") == "api":
                docs = [d for d, _ in api_rerank(case["question"], docs, config.get("k", 5))]
            answer = self.generator.generate(case["question"], docs)
            faithfulness = evaluate_faithfulness(self.llm, "\n".join(docs), answer)
            relevance = evaluate_relevance(self.llm, case["question"], answer)
        return {"avg_faithfulness": ..., "avg_relevance": ..., "avg_overall": ..., "results": [...]}

    def grid_search(self, test_cases: list) -> list:
        configs = [
            {"retriever": "vector", "k": 3, "rerank": "none"},
            {"retriever": "vector", "k": 5, "rerank": "none"},
            {"retriever": "vector", "k": 5, "rerank": "api"},
            {"retriever": "hyde", "k": 5, "rerank": "none"},
            {"retriever": "multi_route", "k": 5, "rerank": "api"},
        ]
```

### main.py — CLI 入口

提供三种运行模式：单次提问、批量实验、网格搜索。

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", type=str, help="提问单个问题")
    parser.add_argument("--config", type=str, default="vector", help="检索策略")
    parser.add_argument("--k", type=int, default=5, help="检索 Top-K")
    parser.add_argument("--rerank", type=str, default="none", help="重排序")
    parser.add_argument("--grid-search", action="store_true", help="网格搜索")

    args = parser.parse_args()

    rag = _init_rag(db_path)  # 从 rag_knowledge 目录加载文档
    runner = ExperimentRunner(rag.store, rag.llm)

    if args.grid_search:
        runner.grid_search(get_test_cases())
    elif args.ask:
        result = runner.run_config([{"question": args.ask}], config)
        print(f"答案: {result['results'][0]['answer']}")
    else:
        result = runner.run_config(get_test_cases(), config)
```

## 使用方法

项目提供了三个主要命令入口，分别用于运行单次实验、交互式问答和自动网格搜索最佳参数。

```bash
# 1. 运行实验（默认配置）
python -m rag_pipeline.main

# 2. 提问单个问题
python -m rag_pipeline.main --ask "RAG 的优势是什么？" --config vector --k 5 --rerank api

# 3. 网格搜索最佳参数
python -m rag_pipeline.main --grid-search
```

## 验收清单

- [ ] 支持多种检索策略（向量 / HyDE / 多路召回）
- [ ] 支持 API Rerank
- [ ] LLM-as-Judge 自动评估
- [ ] 网格搜索最佳配置组合
- [ ] 输出实验报告（可导出 JSON）
- [ ] 可对比不同配置的效果差异

> 评估方法详见 P3（检索质量）和 P4（生成质量）；上下文注入策略原理详见 Prompt D2-P1

## 下一步 → Day5 进阶RAG架构