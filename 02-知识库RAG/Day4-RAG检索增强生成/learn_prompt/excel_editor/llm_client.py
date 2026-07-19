import os, json, time, logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-v4-flash"
        self.logger = logging.getLogger(__name__)

    def chat(self, messages, tools=None, tool_choice="auto", response_format=None):
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            self.logger.error(f"API 调用失败: {e}")
            raise

    def chat_text(self, messages):
        response = self.chat(messages)
        return response.choices[0].message.content

    def chat_json(self, messages):
        response = self.chat(messages, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
