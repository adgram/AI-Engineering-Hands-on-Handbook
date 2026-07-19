"""测试集管理"""

TEST_CASES = [
    {"question": "RAG 是什么？它的工作流程是怎样的？"},
    {"question": "Rerank 重排序模型有什么作用？"},
    {"question": "HyDE 方法如何改善检索效果？"},
    {"question": "如何评估 RAG 系统的质量？"},
]


def get_test_cases() -> list:
    return TEST_CASES
