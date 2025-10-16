import asyncio
import itertools
import re
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APITimeoutError, InternalServerError
from stage_2_generate.config.api_keys import API_KEYS
from stage_2_generate.config.settings import DEFAULT_MODEL, DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY

class APICaller:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        retry_attempts_per_key: int = DEFAULT_RETRY_ATTEMPTS,  
        retry_delay: int = DEFAULT_RETRY_DELAY,
        max_keys_to_try: int = 20  
    ):
        self.model = model
        self.retry_attempts_per_key = retry_attempts_per_key
        self.retry_delay = retry_delay
        self.max_keys_to_try = max_keys_to_try or len(API_KEYS)
        
        # 判断模型类型并设置相应的base_url
        self.is_gpt_model = self._is_gpt_model(model)
        if self.is_gpt_model:
            # GPT系列使用美团内部API
            base_url = "https://aigc.sankuai.com/v1/openai/native"
            self.print(f"[APICaller] Detected GPT model: {model}, using base_url: {base_url}")
        else:
            # Claude系列使用美团内部API
            base_url = "https://aigc.sankuai.com/v1/openai/native"
            self.print(f"[APICaller] Detected Claude model: {model}, using base_url: {base_url}")

        self.api_keys_info = [
            {
                "key": key,
                "client": AsyncOpenAI(api_key=key, base_url=base_url),
                "count": 0
            }
            for key in API_KEYS
        ]
        self.client_cycler = itertools.cycle(self.api_keys_info)
    
    def _is_gpt_model(self, model: str) -> bool:
        """判断是否为GPT系列模型"""
        gpt_patterns = [
            'gpt-3.5',
            'gpt-4',
            'gpt-4-turbo',
            'gpt-4o',
            'gpt-4.1'
        ]
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in gpt_patterns)
    
    async def generate(self, messages: list, max_tokens: int = 8192, temperature: float = 0.0) -> str:
        keys_tried = 0
        max_keys = min(self.max_keys_to_try, len(API_KEYS))
        
        for key_attempt in range(max_keys):
            client_wrapper = next(self.client_cycler)
            client = client_wrapper["client"]
            keys_tried += 1
            
            self.print(f"[APICaller] Trying API key ...{client_wrapper['key'][-4:]} (key {keys_tried}/{max_keys})")
            
            for attempt in range(self.retry_attempts_per_key):
                try:
                    client_wrapper["count"] += 1

                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    response_txt = response.choices[0].message.content
                    self.print(f"[APICaller] Success with key ...{client_wrapper['key'][-4:]} on attempt {attempt + 1}")
                    return response_txt
                    
                except (
                    APIConnectionError,
                    RateLimitError,
                    APITimeoutError,
                    InternalServerError,
                ) as e:
                    self.print(
                        f"[APICaller] API call for key ...{client_wrapper['key'][-4:]} failed on attempt {attempt + 1}/{self.retry_attempts_per_key}. Error: {e}"
                    )
                    if attempt < self.retry_attempts_per_key - 1:
                        self.print(f"[APICaller] Retrying same key in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                        
                except Exception as e:
                    self.print(
                        f"[APICaller] Unexpected error for key ...{client_wrapper['key'][-4:]} on attempt {attempt + 1}/{self.retry_attempts_per_key}. Error: {e}"
                    )
                    if attempt < self.retry_attempts_per_key - 1:
                        self.print(f"[APICaller] Retrying same key in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
            
            self.print(f"[APICaller] Key ...{client_wrapper['key'][-4:]} failed after {self.retry_attempts_per_key} attempts.")

            if key_attempt < max_keys - 1:
                self.print(f"[APICaller] Switching to next API key in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)
        
        self.print(f"[APICaller] All {keys_tried} API keys failed after {self.retry_attempts_per_key} attempts each.")
        return None
    def print(self, *args, **kwargs):
        print(*args, **kwargs)
