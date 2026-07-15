class InferenceBudget:
    def __init__(self, max_rounds: int = 10, max_tokens: int = 32000):
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.rounds_used = 0
        self.tokens_used = 0

    def can_continue(self) -> bool:
        return (
            self.rounds_used < self.max_rounds and
            self.tokens_used < self.max_tokens
        )

    def record_round(self, tokens: int):
        self.rounds_used += 1
        self.tokens_used += tokens

    def summary(self) -> str:
        return (f"预算: {self.rounds_used}/{self.max_rounds} 轮, "
                f"{self.tokens_used}/{self.max_tokens} tokens")
