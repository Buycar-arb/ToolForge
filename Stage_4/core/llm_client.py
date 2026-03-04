"""LLM Client"""

import aiohttp
import asyncio
import itertools
import random
from typing import List, Dict, Optional


class AsyncLLMGenerateLabel:
    """Asynchronous LLM label generator"""
    
    def __init__(self, api_keys: List[str], api_url: str, model: str, temperature: int, max_tokens: int):
        self.api_keys = api_keys
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key_cycle = itertools.cycle(self.api_keys)
        
    def get_next_api_key(self) -> str:
        """Retrieve the next API key in rotation"""
        return next(self.api_key_cycle)
    
    async def call_llm_api(self, session: aiohttp.ClientSession, messages: List[Dict[str, str]], 
                          max_retries: int = 5) -> Optional[str]:
        """Call the LLM API asynchronously"""
        for attempt in range(max_retries):
            try:
                current_api_key = self.get_next_api_key()
                headers = {
                    'Authorization': f'Bearer {current_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    return response_json['choices'][0]['message']['content']
                    
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:50]}...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 + random.uniform(0, 1))
        
        return None
