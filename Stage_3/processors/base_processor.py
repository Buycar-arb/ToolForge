 # Basic Processor
from abc import ABC, abstractmethod
import copy
import json
import random
import re
import ast
import aiohttp
from stage_2_generate.utils.text_utils import extract_tags_as_str_list, parse_jsonl_string
from stage_2_generate.utils.bm25_utils import BM25Processor
from stage_2_generate.utils.file_utils import FileProcessor

class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        # Basic configs
        self.system_prompt = kwargs.get('system_prompt')
        self.tool_list = kwargs.get('random_all_tools_content') # All tools in tool_prompt except the golden tool
        self.tool_prompt = kwargs.get('tool_prompt')
        self.good_tool_content = kwargs.get('good_tool_content')
        self.user_prompt = kwargs.get('user_prompt')
        self.general_tool_name = kwargs.get('general_tool_name')
        self.tool_prompt_general = kwargs.get('tool_prompt_general')
        self.tool_list_general = kwargs.get('random_all_tools_content_general')
        self.good_tool_mapping = kwargs.get('good_tool_mapping')
        # Check whether the query-related tool is included in the system_prompt
        self.simulate_recall_tools_json = kwargs.get('simulate_recall_tools_json')
        self.simulate_recall_tools_general_json = kwargs.get('simulate_recall_tools_general_json')
        # Query-related settings
        self.query = kwargs.get('query')
        self.answer = kwargs.get('answer')
        self.reasoning = kwargs.get('reasoning')
        self.wheel_type = kwargs.get('wheel_type')
        
        # Model parameters
        self.model = kwargs.get('model')
        self.max_tokens = kwargs.get('max_tokens')
        self.temperature = kwargs.get('temperature')
        
        # Data-related settings
        self.gold_contents = kwargs.get('gold_contents')
        self.all_contents = kwargs.get('all_contents')
        
        # API calling
        self.call_llm_api = kwargs.get('call_llm_api')
        self.call_claude_api = kwargs.get('call_claude_api')
        
    
        self.bm25_processor = BM25Processor()
        self.file_processor = FileProcessor()
        
        self.type = kwargs.get('type', self.wheel_type)  
    
    
    def extract_tags_as_str_list(self, text, tag, return_as_list=True):
        return extract_tags_as_str_list(text, tag, return_as_list)
    
    def parse_jsonl_string(self, data_string):
        return parse_jsonl_string(data_string)
    
    def bm25s_function(self, corpus, query, top_k_min, top_k_max, language='english'):
        return self.bm25_processor.bm25s_function(corpus, query, top_k_min, top_k_max, language)
    
    def deduplicate_rag_results(self, nested_list):

        if not nested_list:
            return []
        
        if isinstance(nested_list[0], list):
            flat_list = []
            for sublist in nested_list:
                flat_list.extend(sublist)
            nested_list = flat_list
        
        unique_items = {}
        for item in nested_list:
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                if content not in unique_items:
                    unique_items[content] = item
        
        result = list(unique_items.values())
        random.shuffle(result)
        return result
    
    def load_tool_definitions(self, tools_dir):
        """Load tool definitions"""
        return self.file_processor.load_tool_definitions(tools_dir)
    
    def get_grouped_tool_calls_hybrid(self, messages_data, tools_dir):
        """Retrieve grouped tool invocations"""

        assistant_messages = [msg for msg in messages_data if msg.get("role") == "assistant"]
        if len(assistant_messages) <= 1:
            print("1")
            return []
        
        # Load all tool definitions
        tool_definitions = self.load_tool_definitions(tools_dir)
        
        result = []
        for assistant_index, assistant_msg in enumerate(assistant_messages[:-1]):
            content = assistant_msg.get("content", "")
            matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
            
            if matches:
                tool_calls = []
                for match in matches:
                    try:
                        tool_call_obj = json.loads(match.strip())
                        
                        # Find the corresponding tool definition and add it to "tool_call_obj"
                        tool_name = tool_call_obj.get('name')
                        if tool_name and tool_name in tool_definitions:
                            tool_call_obj['tool_definition'] = tool_definitions[tool_name]
                        else:
                            tool_call_obj['tool_definition'] = None
                            print(f"Warning: Tool definition not found: {tool_name}")
                        
                        tool_calls.append(tool_call_obj)
                    
                    except json.JSONDecodeError:
                        print(f"Error: JSON parsing failed.: {match}")
                        continue
                
                assistant_result = {
                    "assistant_index": assistant_index + 1,
                    "objects": tool_calls,
                }
                
                result.append(assistant_result)
        
        return result
    

    @abstractmethod
    async def process(self, processor_case):
        if processor_case not in self.processors:
            raise ValueError(f"Unsupported processing type: {processor_case}")
        return await self.processors[processor_case]()