import time, logging


class TaskExecutor:
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.success_count = 0
        self.failure_count = 0
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TaskExecutor")

    def execute(self, task, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                result = task(*args, **kwargs)
                self.success_count += 1
                self.logger.info(f"任务成功 (尝试{attempt+1})")
                return result
            except Exception as e:
                self.logger.warning(f"任务失败 (尝试{attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        self.failure_count += 1
        raise Exception(f"任务在{self.max_retries}次尝试后仍然失败")

    def stats(self):
        return {
            "success": self.success_count,
            "failure": self.failure_count,
            "success_rate": self.success_count / max(
                self.success_count + self.failure_count, 1
            )
        }
