import json
from typing import List, Dict, Any, Tuple
import itertools
import asyncio
import uuid
import os
import sys
from datetime import datetime
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
)

# API Configuration
model = "gpt-4.1"
temperature = 0.0
max_tokens = 8192
# For standalone testing of this script
# from tool_prompts import TOOL_CHOOSE_USER_BCD
# from tool_prompts import TOOL_CHOOSE_SYSTEM_BCD
# Combined with gradio
from stage_1_label.code.tool_prompts import TOOL_CHOOSE_USER_BCD
from stage_1_label.code.tool_prompts import TOOL_CHOOSE_SYSTEM_BCD
from stage_1_label.code.tool_prompts import TOOL_LIST

# API Keys Configuration
# Option 1: Load from environment variable
api_keys_env = os.getenv("API_KEYS", "")
if api_keys_env:
    api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
else:
    # Option 2: Load from file or use default empty list
    # You should set your API keys via environment variable or configuration file
    api_keys = [
        "your-api-key-1",
        "your-api-key-2",
        "your-api-key-3",
        "your-api-key-4",
        "your-api-key-5",
        "your-api-key-6",
        "your-api-key-7",
        "your-api-key-8",
        "your-api-key-9",
        "your-api-key-10",
        "your-api-key-11",
        "your-api-key-12",
        "your-api-key-13",
        "your-api-key-14",
        "your-api-key-15",
        "your-api-key-16",
        "your-api-key-17",
        "your-api-key-18",
        "your-api-key-19",
        "your-api-key-20",
        "your-api-key-21",
        "your-api-key-22"
    ]

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

        # API Base URL Configuration
        API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        
        self.api_keys_info = [
            {
                "key": key,
                "client": AsyncOpenAI(api_key=key, base_url=API_BASE_URL),
                "count": 0
            }
            for key in api_keys
        ]
        self.client_cycler = itertools.cycle(self.api_keys_info)
        # Add lock to protect thread safety of client_cycler
        self._lock = asyncio.Lock()
    
    async def generate(self, messages: list, system_prompt: str = None) -> str:
        # Thread-safely get the next client
        async with self._lock:
            client_wrapper = next(self.client_cycler)
        
        client = client_wrapper["client"]

        for attempt in range(self.retry_attempts):
            try:
                if attempt == 0:
                    async with self._lock:
                        client_wrapper["count"] += 1

                # If there is a system_prompt, add it to the beginning of messages
                final_messages = messages.copy()
                if system_prompt:
                    final_messages.insert(0, {"role": "system", "content": system_prompt})

                response = await client.chat.completions.create(
                    model=self.model,
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response_txt = response.choices[0].message.content
                return response_txt
                
            except (
                APIConnectionError,
                RateLimitError,
                APITimeoutError,
                InternalServerError,
            ) as e:
                self.print(
                    f"[APICaller] API call for key ...{client_wrapper['key'][-4:]} failed on attempt {attempt + 1}/{self.retry_attempts}. Error: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    self.print(f"[APICaller] Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.print("[APICaller] API call failed after all retries.")
                    return None
            except Exception as e:
                self.print(
                    f"[APICaller] An unexpected, non-retryable error occurred: {e}"
                )
                return None
        return None
    
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

class LLMGenerateLabel:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize APICaller
        self.api_caller = APICaller(model=model, retry_attempts=5, retry_delay=10)
    
    async def call_api(self, messages: list, system_prompt: str) -> dict:
        """Use APICaller to make API calls"""
        try:
            content = await self.api_caller.generate(messages, system_prompt)
            if content:
                return {
                    "content": content
                }
            else:
                return None
        except Exception as e:
            print(f"API call failed: {e}")
            return None

async def process_single_line(generator_label, line_data, generate_count, content_callback=None):
    """Function to process single data item with retry mechanism"""
    max_retries = 5
    
    # Generate a processing ID for this data item
    process_id = f"line_{generate_count}_{str(uuid.uuid4())[:8]}"
    
    for retry_count in range(max_retries):
        try:
            print(f"🚀 Starting processing data item {generate_count} - ProcessId: {process_id}, retry {retry_count + 1}")
            
            # Prepare data
            supporting_facts = line_data["supporting_facts"]
            context = line_data["context"]
            query_type = line_data["type"]
            support_set = set()
            for title, sent_id in supporting_facts:
                support_set.add((title, sent_id))
            gold_contents = [] 
            for title, sentences in context:
                for sent_id, sentence in enumerate(sentences):
                    if (title, sent_id) in support_set:
                        gold_contents.append(sentence)

            user_prompt = TOOL_CHOOSE_USER_BCD.format(
                question = line_data["question"],
                tool_list = TOOL_LIST
            )
            
            # Build message format
            messages = [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            system_prompt = TOOL_CHOOSE_SYSTEM_BCD
            
            # Use new API call method
            general_search_tool = await generator_label.call_api(messages, system_prompt)
            if not general_search_tool or not general_search_tool.get("content"):
                raise Exception("API returned empty content")
            
            # Parse returned content
            content = general_search_tool["content"]
            
            # If callback function is provided, call it to display content
            if content_callback:
                await content_callback(generate_count, content, line_data["question"])
            
            # Check if necessary separators are included
            if "<think>\n" not in content:
                raise Exception("Missing <think> tag in returned content")
            
            temp_one = content.split("<think>\n")
            if len(temp_one) < 2:
                raise Exception("Cannot correctly split <think> content")
            
            if "\n</think>" not in temp_one[1]:
                raise Exception("Missing </think> tag in returned content")
                
            line_think = temp_one[1].split("\n</think>")
            line_data["reasoning"] = line_think[0]
            
            if "\n工具选择:" not in line_think[1]:
                raise Exception("Missing '工具选择:' tag in returned content")
                
            temp_two = line_think[1].split("\n工具选择:")
            if len(temp_two) < 2:
                raise Exception("Cannot correctly split tool selection content")
            
            if "\n路径选择:" not in temp_two[1]:
                raise Exception("Missing '路径选择:' tag in returned content")
                
            tool_select = temp_two[1].split("\n路径选择:")
            if len(tool_select) < 2:
                raise Exception("Cannot correctly split route selection content")
                
            line_data["tool_select"] = tool_select[0]
            line_data["route_select"] = tool_select[1]
            
            print(f"✅ Data item {generate_count} processed successfully - ProcessId: {process_id}")
            return line_data
            
        except Exception as e:
            print(f"❌ Data item {generate_count} processing failed - ProcessId: {process_id}, retry {retry_count + 1}, error: {e}")
            if retry_count < max_retries - 1:
                print(f"⏳ Waiting 5 seconds before retry... ProcessId: {process_id}")
                await asyncio.sleep(5)
            else:
                print(f"💀 Data item {generate_count} failed after {max_retries} retries - ProcessId: {process_id}")
                # Return original data with error markers
                line_data["processing_error"] = f"Processing failed: {str(e)}"
                line_data["reasoning"] = "Processing failed"
                line_data["tool_select"] = "Processing failed"
                line_data["route_select"] = "Processing failed"
                return line_data

async def main():
    # Configuration parameters
    MAX_LINES = 5000  # Only process first 5000 lines
    CONCURRENCY = 10  # Concurrency count
    
    generator_label = LLMGenerateLabel(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    input_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_5.jsonl"
    output_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_1.jsonl"
    residue_file = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/chenhao/ToolForge_github/code_perfect/data/original_data/test_2.jsonl"
    
    # Read data and separate processed and remaining data
    processed_data = []
    residue_data = []
    generate_count = 1
    
    print(f"📚 Starting to read data, first {MAX_LINES} lines for processing, remaining lines saved to residue file...")
    
    with open(input_file, "r") as f:
        for line in f:
            try:
                line_data = json.loads(line)
                
                if generate_count <= MAX_LINES:
                    # First MAX_LINES lines for processing
                    processed_data.append((line_data, generate_count))
                else:
                    # Remaining lines saved to residue_data
                    residue_data.append(line_data)
                
                generate_count += 1
            except json.JSONDecodeError as e:
                print(f"Data item {generate_count} JSON parsing failed: {e}")
                generate_count += 1
                continue
    
    print(f"📚 Data reading completed:")
    print(f"   Data for processing: {len(processed_data)} items")
    print(f"   Remaining data: {len(residue_data)} items")
    print(f"   Total data: {generate_count - 1} items")
    
    # Save remaining data to residue file
    if residue_data:
        print(f"💾 Starting to save remaining data to {residue_file}...")
        with open(residue_file, "w", encoding='utf-8') as rf:
            for residue_item in residue_data:
                rf.write(json.dumps(residue_item, ensure_ascii=False) + '\n')
        print(f"✅ {len(residue_data)} remaining data items saved to: {residue_file}")
    else:
        print("ℹ️  No remaining data to save")
    
    # If no data to process, exit directly
    if not processed_data:
        print("⚠️  No data to process, program ends")
        return
    
    print(f"🚀 Starting concurrent processing of {len(processed_data)} data items, concurrency: {CONCURRENCY}")
    
    # Create semaphore to control concurrency count
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def process_with_semaphore(data_item):
        async with semaphore:
            line_data, count = data_item
            return await process_single_line(generator_label, line_data, count)
    
    # Process all data concurrently while maintaining original order
    print("⏳ Starting concurrent processing...")
    tasks = [process_with_semaphore(data_item) for data_item in processed_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("💾 Starting to save processing results in order...")
    
    # Save results in order
    success_count = 0
    error_count = 0
    
    with open(output_file, "w", encoding='utf-8') as w:
        for i, result in enumerate(results):
            original_count = processed_data[i][1]  # Get original line number
            
            if isinstance(result, Exception):
                print(f"❌ Data item {original_count} processing exception: {result}")
                # Create error result
                line_data = processed_data[i][0]
                line_data["processing_error"] = f"Processing exception: {str(result)}"
                line_data["reasoning"] = "Processing exception"
                line_data["tool_select"] = "Processing exception"
                line_data["route_select"] = "Processing exception"
                processed_line = line_data
                error_count += 1
            else:
                processed_line = result
                success_count += 1
            
            # Write results in order
            w.write(json.dumps(processed_line, ensure_ascii=False) + '\n')
            
            # Show progress every 100 items
            if (i + 1) % 100 == 0:
                print(f"📝 Saved {i + 1}/{len(results)} processing results")
    
    print(f"\n📊 Processing completion statistics:")
    print(f"   Processed data items: {len(processed_data)}")
    print(f"   Success count: {success_count}")
    print(f"   Failure count: {error_count}")
    print(f"   Remaining data items: {len(residue_data)}")
    print(f"   Concurrency: {CONCURRENCY}")
    print(f"   Processing results saved to: {output_file}")
    print(f"   Remaining data saved to: {residue_file}")

if __name__ == "__main__":
    asyncio.run(main())
