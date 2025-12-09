"""Closed source model implementations (GPT-4, DeepSeek)."""

import os
import json
import time
import asyncio
import itertools
import re
from typing import Dict, List, Any
import requests
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)
from .base_model import BaseModel


# API Keys for the new implementation
API_KEYS = [
    "1801520868460679207",
    "1801520829835173916",
    "1889146223668690996",
    "1920105122387161134",
    "1920105181459755034",
    "1718871834076307489",
    "1737820362802688060",
    "1742879892716675171",
    "1775083086435561521",
    "1796110759362199605",
    "1794984524074676265",
    "1792464990918938657",
    "1789931624457855004",
    "1789903374117359714",
    "1787822145012584505",
    "1785146944130732056",
    "1785146918596018242",
    "1785146889328074779",
    "1785146861184548942",
    "1785146832369369093",
    "1783462889744248849",
    "1783462795028553741"
]


def find_closing_quote(text, start_pos):
    """
    找到字符串中对应的结束引号，考虑转义
    """
    i = start_pos
    while i < len(text):
        if text[i] == '"':
            # 检查是否被转义
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == '\\':
                backslash_count += 1
                j -= 1
            
            # 如果反斜杠数量是偶数，说明引号没有被转义
            if backslash_count % 2 == 0:
                return i
        i += 1
    return -1


def extract_messages_brutal(text):
    """
    暴力解析：直接提取role和content，然后重新构建JSON
    """
    try:
        # 1. 提取JSON代码块
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if not match:
            print("未找到JSON代码块")
            return None
        
        json_content = match.group(1).strip()
        
        # 2. 暴力提取每个message对象
        messages = []
        
        # 简单方法：按 "role": 分割
        parts = json_content.split('"role":')
        
        for i, part in enumerate(parts):
            if i == 0:  # 第一部分通常是 { "messages": [
                continue
                
            # 提取role值
            role_match = re.search(r'^\s*"([^"]+)"', part)
            if not role_match:
                continue
            role = role_match.group(1)
            
            # 提取content值 - 这里是关键
            # 寻找 "content": "..." 但要处理多行和转义
            content_pattern = r'"content":\s*"(.*?)"\s*(?=\}|,\s*\})'
            content_match = re.search(content_pattern, part, re.DOTALL)
            
            if content_match:
                content = content_match.group(1)
                content = (content.replace('\\n', '\n')
                                 .replace('\\r', '\r')
                                 .replace('\\t', '\t')
                                 .replace('\\"', '"')
                                 .replace('\\\\', '\\'))
                
                # 不需要转义，直接使用原始内容
                messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # 如果正则匹配失败，尝试手动查找
                content_start = part.find('"content":')
                if content_start != -1:
                    # 找到content后的引号
                    quote_start = part.find('"', content_start + len('"content":'))
                    if quote_start != -1:
                        # 找到对应的结束引号（考虑转义）
                        quote_end = find_closing_quote(part, quote_start + 1)
                        if quote_end != -1:
                            content = part[quote_start + 1:quote_end]
                            messages.append({
                                "role": role,
                                "content": content
                            })
        
        return messages
        
    except Exception as e:
        print(f"暴力解析失败: {e}")
        return None


def merge_messages(parsed_messages, original_system_message):
    """
    合并原始消息和解析出的消息
    
    Args:
        parsed_messages: 从模型输出解析出的消息列表
        original_system_message: 原始的system消息
    
    Returns:
        合并后的完整消息列表
    """
    try:
        final_messages = []
        
        # 1. 添加原始system消息
        if original_system_message:
            final_messages.append(original_system_message)
        
        # 3. 添加解析出的新消息（通常是assistant和tool消息）
        if parsed_messages:
            # 过滤掉可能重复的user消息（如果解析结果中包含）
            for msg in parsed_messages:
                # 只添加assistant和tool类型的消息，避免重复user消息
                if msg.get('role') in ['user', 'assistant', 'tool']:
                    final_messages.append(msg)
        
        return final_messages
        
    except Exception as e:
        print(f"消息合并失败: {e}")
        return None


class APICaller:
    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4",
        retry_attempts: int = 5,
        retry_delay: int = 10
    ):
        self.model = model
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self.api_keys_info = [
            {
                "key": key,
                "client": AsyncOpenAI(api_key=key, base_url="https://aigc.sankuai.com/v1/openai/native"),
                "count": 0
            }
            for key in API_KEYS
        ]
        self.client_cycler = itertools.cycle(self.api_keys_info)
        self._lock = asyncio.Lock()
    
    async def generate(self, messages: list) -> str:
        async with self._lock:
            client_wrapper = next(self.client_cycler)
        
        client = client_wrapper["client"]

        for attempt in range(self.retry_attempts):
            try:
                if attempt == 0:
                    async with self._lock:
                        client_wrapper["count"] += 1

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=10000,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except (APIConnectionError, RateLimitError, APITimeoutError, InternalServerError) as e:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    return None
            except Exception as e:
                return None
        return None


class LLMGenerator:
    def __init__(self, model: str = "anthropic.claude-sonnet-4", retry_attempts: int = 15, retry_delay: int = 60):
        self.api_caller = APICaller(model=model, retry_attempts=retry_attempts, retry_delay=retry_delay)
    
    async def call_api(self, messages: list) -> str:
        return await self.api_caller.generate(messages)


class OpenAIModel(BaseModel):
    """OpenAI GPT-4 implementation with new async approach."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)
        # Initialize the new async generator
        self.llm_generator = LLMGenerator(
            model=self.model_name,
            retry_attempts=15,
            retry_delay=60
        )

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Put everthing in prompt (模仿raw text)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,  # Use the chat completions endpoint
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                # The API strips stop sequences, so we need to add them back
                if stop_sequences and content:
                    # Check for each possible unclosed tag and append the appropriate closing
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling using new async approach."""
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._generate_with_functions_async(messages, tools, **kwargs))
            return result
        finally:
            loop.close()

    async def _generate_with_functions_async(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Async implementation of function calling."""
        try:
            # Call the API using the new async approach
            result = await self.llm_generator.call_api(messages)
            
            if result:
                # For the evaluation framework, we need to return the result directly
                # The result should be the model's response content
                return {
                    'content': result,
                    'tool_calls': []
                }
            
            # Fallback if no result
            return {
                'content': '',
                'tool_calls': []
            }
            
        except Exception as e:
            print(f"Error in async function calling: {e}")
            return {
                'content': '',
                'tool_calls': []
            }


class ClaudeModel(BaseModel):
    """Claude model implementation with thinking capability and new async approach."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)
        self.thinking = config.get('thinking', {})
        # Initialize the new async generator
        self.llm_generator = LLMGenerator(
            model=self.model_name,
            retry_attempts=15,
            retry_delay=60
        )

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Put everything in prompt (模仿raw text)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences
        }

        # Add thinking configuration if available
        if self.thinking:
            data["thinking"] = self.thinking

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                if stop_sequences and content:
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling using new async approach."""
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._generate_with_functions_async(messages, tools, **kwargs))
            return result
        finally:
            loop.close()

    async def _generate_with_functions_async(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Async implementation of function calling for Claude."""
        try:
            # Call the API using the new async approach
            result = await self.llm_generator.call_api(messages)
            
            if result:
                # For the evaluation framework, we need to return the result directly
                # The result should be the model's response content
                return {
                    'content': result,
                    'tool_calls': []
                }
            
            # Fallback if no result
            return {
                'content': '',
                'tool_calls': []
            }
            
        except Exception as e:
            print(f"Error in async function calling: {e}")
            return {
                'content': '',
                'tool_calls': []
            }


class GrokModel(BaseModel):
    """Grok model implementation with new async approach."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)
        # Initialize the new async generator
        self.llm_generator = LLMGenerator(
            model=self.model_name,
            retry_attempts=15,
            retry_delay=60
        )

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Put everything in prompt (模仿raw text)
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences,
            "stream": False  # Disable streaming for tag-based generation
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                if stop_sequences and content:
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling using new async approach."""
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._generate_with_functions_async(messages, tools, **kwargs))
            return result
        finally:
            loop.close()

    async def _generate_with_functions_async(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Async implementation of function calling for Grok."""
        try:
            # Call the API using the new async approach
            result = await self.llm_generator.call_api(messages)
            
            if result:
                # For the evaluation framework, we need to return the result directly
                # The result should be the model's response content
                return {
                    'content': result,
                    'tool_calls': []
                }
            
            # Fallback if no result
            return {
                'content': '',
                'tool_calls': []
            }
            
        except Exception as e:
            print(f"Error in async function calling: {e}")
            return {
                'content': '',
                'tool_calls': []
            }


class DeepSeekModel(BaseModel):
    """DeepSeek model implementation with new async approach."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key'] if not config['api_key'].startswith('${') else os.getenv(config['api_key'].replace('${', '').replace('}', ''))
        self.endpoint = config['endpoint']
        self.model_name = config['model_name']
        self.timeout = config.get('timeout', 60)
        # Initialize the new async generator
        self.llm_generator = LLMGenerator(
            model=self.model_name,
            retry_attempts=15,
            retry_delay=60
        )

    def generate_with_tags(self, prompt: str, stop_sequences: List[str] = None, **kwargs) -> str:
        """Generate response using chat completions API with stop sequences."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Use chat completions endpoint with messages format
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
            "stop": stop_sequences
        }

        for retry in range(3):
            try:
                response = requests.post(
                    self.endpoint,  # Use the chat completions endpoint
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()

                # Get the response content
                content = response.json()['choices'][0]['message']['content']

                # Append the stop sequence that was triggered
                # The API strips stop sequences, so we need to add them back
                if stop_sequences and content:
                    # Check for each possible unclosed tag and append the appropriate closing
                    if '<search>' in content and '</search>' not in content:
                        content += '</search>'
                    elif '<answer>' in content and '</answer>' not in content:
                        content += '</answer>'

                return content
            except Exception as e:
                if retry == 2:
                    raise e
                time.sleep(2 ** retry)

    def generate_with_functions(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Generate response with function/tool calling using new async approach."""
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._generate_with_functions_async(messages, tools, **kwargs))
            return result
        finally:
            loop.close()

    async def _generate_with_functions_async(self, messages: List[Dict[str, str]], tools: List[Dict], **kwargs) -> Dict:
        """Async implementation of function calling for DeepSeek."""
        try:
            # Call the API using the new async approach
            result = await self.llm_generator.call_api(messages)
            
            if result:
                # For the evaluation framework, we need to return the result directly
                # The result should be the model's response content
                return {
                    'content': result,
                    'tool_calls': []
                }
            
            # Fallback if no result
            return {
                'content': '',
                'tool_calls': []
            }
            
        except Exception as e:
            print(f"Error in async function calling: {e}")
            return {
                'content': '',
                'tool_calls': []
            }
