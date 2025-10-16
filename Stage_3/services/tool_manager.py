# 工具管理服务
import random
import re
import json
from stage_2_generate.prompts.tool_prompt_template import tool_prompt_new
from stage_2_generate.utils.file_utils import FileProcessor
from stage_2_generate.config.settings import TARGET_GENERAL_FILE
from pathlib import Path
import copy
class ToolManager:
    def __init__(self):
        pass
    
    def load_prompts(self, system_prompt_file, user_prompt_file, tool_bank_file, good_tool, vritual_tool_number_min, vritual_tool_number_max):
    
        new_random_all_tools_except_good_tool_general = []
        simulate_recall_tools_general_json = []
        general_tool_name_and_content = ""
        tool_prompt_general = ""
        
        with open(system_prompt_file, 'r') as file:
            system_prompt = file.read()
        with open(user_prompt_file, 'r') as file:
            user_prompt = file.read()
       
        random_all_tools, good_tool_content, available_general_status, good_general_status, general_tool_name_and_content_from_files, good_tool_mapping = FileProcessor.load_random_tools_excluding_good_tools(tool_bank_file, good_tool)

        vritual_tool_number = random.randint(vritual_tool_number_min, vritual_tool_number_max)
        if good_general_status == "exist general_file":
            tool_prompt_general = "error"
            general_tool_name_and_content = "error"
        else:
            tool_general = FileProcessor._extract_random_tool_from_file(TARGET_GENERAL_FILE)
            
            random_tool_general = random.sample(random_all_tools, min(vritual_tool_number, len(random_all_tools)))
            random_tool_general.extend(good_tool_content)
            if available_general_status == "exist general_file":
                random_tool_general.append(general_tool_name_and_content_from_files)
                general_tool_name_and_content = general_tool_name_and_content_from_files
            else:
                random_tool_general.append(tool_general)  
                general_tool_name_and_content = tool_general
            
            random.shuffle(random_tool_general)
            new_random_all_tools_except_good_tool_general = copy.deepcopy(random_tool_general)
            simulate_recall_tools_general = ''
            simulate_recall_tools_general_json = []
            for every_tool in random_tool_general:
                simulate_recall_tools_general_json.append(every_tool)
                simulate_recall_tools_general += str(every_tool)
            tool_prompt_general = tool_prompt_new.substitute(recall_tools=simulate_recall_tools_general)
        random_tool = random.sample(random_all_tools, min(vritual_tool_number, len(random_all_tools)))
        new_random_all_tools_except_good_tool = copy.deepcopy(random_tool)
        random_tool.extend(good_tool_content) 
        random.shuffle(random_tool)
        simulate_recall_tools = ''
        simulate_recall_tools_json = []
        for every_tool in random_tool:
            simulate_recall_tools_json.append(every_tool)
            simulate_recall_tools += str(every_tool)
        tool_prompt = tool_prompt_new.substitute(recall_tools = simulate_recall_tools)

        return new_random_all_tools_except_good_tool, system_prompt, tool_prompt, good_tool_content, user_prompt, general_tool_name_and_content, tool_prompt_general, new_random_all_tools_except_good_tool_general, good_tool_mapping, simulate_recall_tools_json, simulate_recall_tools_general_json
    def get_grouped_tool_calls_hybrid(self, messages_data, tools_dir):
       

        assistant_messages = [msg for msg in messages_data if msg.get("role") == "assistant"]
        
        if len(assistant_messages) <= 1:
            print("1")
            return []
        
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
                        
                        tool_name = tool_call_obj.get('name')
                        if tool_name and tool_name in tool_definitions:
                            tool_call_obj['tool_definition'] = tool_definitions[tool_name]
                        else:
                            tool_call_obj['tool_definition'] = None
                            print(f"Warning: Tool definition not found: {tool_name}")
                        
                        tool_calls.append(tool_call_obj)
                    
                    except json.JSONDecodeError:
                        print(f"JSON parsing failed: {match}")
                        continue
                
                assistant_result = {
                    "assistant_index": assistant_index + 1,
                    "objects": tool_calls,  
                }
                
                result.append(assistant_result)
        
        return result
    
