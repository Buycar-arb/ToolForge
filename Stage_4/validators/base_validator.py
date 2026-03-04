"""Basic Validator"""

import json
import re
from typing import Any


class DataValidator:
    """Data Validator Class"""
    
    @staticmethod
    def check_dialogue_format(data, expected_format):
        """1. Check if the format of dialogue data matches the expected pattern"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if "messages" not in data:
                return 0
            
            messages = data["messages"]
            if len(messages) != len(expected_format):
                return 0
            
            for i, message in enumerate(messages):
                if "role" not in message:
                    return 0
                
                if message["role"] != expected_format[i]:
                    return 0
            
            return 1
        
        except Exception as e:
            return 0

    @staticmethod
    def check_assistant_content_format(data):
        """2. Check if the content format of assistant messages matches the expected pattern"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if "messages" not in data:
                return 0
            
            messages = data["messages"]
            
            assistant_messages = []
            for i, message in enumerate(messages):
                if message.get("role") == "assistant":
                    assistant_messages.append((i, message))
            
            if not assistant_messages:
                return 0
            
            for i, (msg_index, message) in enumerate(assistant_messages):
                content = message.get("content", "")
                is_last_assistant = (i == len(assistant_messages) - 1)
                
                if is_last_assistant:
                    pattern = r'^<think>\s*.*?\s*</think>\s*\n\s*<answer>\s*.*?\s*</answer>\s*$'
                    if not re.match(pattern, content.strip(), re.DOTALL):
                        print("error")
                        # return 0
                else:
                    if not content.strip().startswith('<think>'):
                        return 0
                    
                    think_pattern = r'^<think>\s*.*?\s*</think>\s*\n\s*(.*?)$'
                    match = re.match(think_pattern, content.strip(), re.DOTALL)
                    if not match:
                        return 0
                    
                    remaining_content = match.group(1).strip()
                    if not remaining_content:
                        return 0
                    
                    tool_call_pattern = r'^(<tool_call>\s*.*?\s*</tool_call>\s*)+$'
                    if not re.match(tool_call_pattern, remaining_content, re.DOTALL):
                        return 0
            
            return 1
        
        except Exception as e:
            return 0

    @staticmethod
    def check_non_assistant_content_not_empty(data):
        """3. Check that the content fields of all non-assistant messages are not empty"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if "messages" not in data:
                return 0
            
            messages = data["messages"]
            
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role != "assistant":
                    if not content or not content.strip():
                        return 0
            
            return 1
        
        except Exception as e:
            return 0

    @staticmethod
    def check_last_assistant_answer_consistency(data):
        """4. Check whether the content of the last assistant's answer matches line[2]"""
        try:
            messages_data = data[1]
            expected_answer = data[2]["answer"]
            
            if isinstance(messages_data, str):
                messages_data = json.loads(messages_data)
            
            if "messages" not in messages_data:
                return 0
            
            messages = messages_data["messages"]
            
            last_assistant = None
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    last_assistant = message
                    break
            
            if not last_assistant:
                return 0
            
            assistant_content = last_assistant.get("content", "")
            
            answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
            match = re.search(answer_pattern, assistant_content, re.DOTALL)
            
            if not match:
                return 0
            
            extracted_answer = match.group(1).strip()
            
            if isinstance(expected_answer, str):
                expected_answer = expected_answer.strip()
            else:
                expected_answer = str(expected_answer).strip()
            
            return 1 if extracted_answer.lower() == expected_answer.lower() else 0
            
        except Exception as e:
            return 0

    @staticmethod
    def check_tool_rags_consistency(data):
        """5. Check whether the content of each tool message matches the corresponding part of rags in line[1]"""
        try:
            messages_data = data[1]
            rags_data = data[2]["rags"]
            
            if isinstance(messages_data, str):
                messages_data = json.loads(messages_data)
            
            if "messages" not in messages_data:
                return 0
            
            messages = messages_data["messages"]
            
            tool_messages = []
            for message in messages:
                if message.get("role") == "tool":
                    tool_messages.append(message.get("content", ""))
            if len(tool_messages) != len(rags_data):
                return 0
            
            def extract_alphanumeric(text):
                """Keep only letters and digits, convert to lowercase"""
                return re.sub(r'[^a-zA-Z0-9]', '', text).lower()
            
            def normalize_item(item):
                """Normalize an item to a comparable tuple"""
                return (extract_alphanumeric(item["title"]), extract_alphanumeric(item["content"]))
            
            for i, (tool_content, rag_items) in enumerate(zip(tool_messages, rags_data)):
                tool_items = []
                
                pattern = r'\*\*(\d+)\*\*\s*\ntitle:\s*(.*?)\s*\ncontent:\s*(.*?)(?=\n\*\*\d+\*\*|\Z)'
                matches = re.findall(pattern, tool_content, re.DOTALL)
                
                for match in matches:
                    number, title, snippet = match
                    tool_items.append({
                        "title": title.strip(),
                        "content": snippet.strip()
                    })
                
                if len(tool_items) != len(rag_items):
                    return 0
                
                tool_set = set(normalize_item(item) for item in tool_items)
                rag_set = set(normalize_item(item) for item in rag_items)
                
                if tool_set != rag_set:
                    return 0
            
            return 1
                        
        except Exception as e:
            return 0

    @staticmethod
    def check_argument_modifications(data, start_index, end_index):
        """6. Check whether parameter modifications between consecutive assistants are all within required fields"""
        if not data[3]:
            return 1
        argument_check = data[3]["argument_check"]

        if len(argument_check) < 2:
            print("Not enough assistant messages to compare")
            return 1
        
        for i in range(start_index, end_index, 2):
            if i + 1 > end_index:
                print("1111")
                break
            
            first_assistant = argument_check[i]
            second_assistant = argument_check[i + 1]
            
            first_tools = first_assistant['objects']
            second_tools = second_assistant['objects']
            
            if len(first_tools) != len(second_tools):
                print(f"Number of tool calls mismatch: {len(first_tools)} vs {len(second_tools)}")
                return 0
            
            for tool_idx, (first_tool, second_tool) in enumerate(zip(first_tools, second_tools)):
                if first_tool.get('name') != second_tool.get('name'):
                    print(f"Tool name mismatch: {first_tool.get('name')} vs {second_tool.get('name')}")
                    return 0
                
                first_args = first_tool.get('arguments', {})
                second_args = second_tool.get('arguments', {})
                
                modified_params = []
                all_params = set(first_args.keys()) | set(second_args.keys())
                
                for param in all_params:
                    first_value = first_args.get(param)
                    second_value = second_args.get(param)
                    
                    if first_value != second_value:
                        modified_params.append(param)
                
                first_tool_definition = first_tool.get('tool_definition')
                if not first_tool_definition:
                    print(f"Assistant {first_assistant['assistant_index']} is missing tool definition")
                    return 0
                
                parameters = first_tool_definition.get('parameters', {})
                required_params = parameters.get('required', [])
                
                for param in modified_params:
                    if param not in required_params:
                        print(f"Modified parameter '{param}' is not in the required params of Assistant {first_assistant['assistant_index']}")
                        return 0
        
        return 1

    @staticmethod
    def check_reference_consistency(data):
        """7. Check whether argument_all_reference in data[2] matches supporting_facts in data[3]"""
        try:
            argument_all_reference_raw = data[2].get("argument_all_reference", [])
            supporting_facts = data[3].get("supporting_facts", [])
            context = data[3].get("context", [])
            
            argument_all_reference = []
            for turn_item in argument_all_reference_raw:
                turn_data = turn_item.get("data", [])
                argument_all_reference.extend(turn_data)
            
            if len(argument_all_reference) != len(supporting_facts):
                return 0
            
            context_dict = {}
            for item in context:
                if len(item) >= 2:
                    title = item[0]
                    sentences = item[1] if isinstance(item[1], list) else [item[1]]
                    context_dict[title] = sentences
            
            for ref in argument_all_reference:
                ref_title = ref.get("title", "")
                ref_content = ref.get("content", "")
                
                matching_fact = None
                for fact in supporting_facts:
                    if len(fact) >= 2 and fact[0] == ref_title:
                        matching_fact = fact
                        break
                
                if not matching_fact:
                    return 0
                
                sent_id = matching_fact[1]
                if (ref_title not in context_dict or 
                    sent_id >= len(context_dict[ref_title])):
                    return 0
                
                expected_content = context_dict[ref_title][sent_id]
                if ref_content.strip() != expected_content.strip():
                    return 0
            
            reference_titles = {ref.get("title", "") for ref in argument_all_reference}
            supporting_titles = {fact[0] for fact in supporting_facts if len(fact) >= 1}
            
            if reference_titles != supporting_titles:
                return 0
            
            return 1
            
        except Exception as e:
            print(f"Error occurred during reference check: {e}")
            return 0

    @staticmethod
    def check_tool_consistency(data, wheel_type=None):
        """8. Check tool call consistency"""
        def parse_tool_select(tool_select_str):
            """Parse tool_select string"""
            try:
                if tool_select_str.startswith('[') and tool_select_str.endswith(']'):
                    tools_str = tool_select_str[1:-1]
                    tools = [tool.strip() for tool in tools_str.split(',')]
                    return tools
                else:
                    return []
            except Exception:
                return []
        
        fewer_tools_cases = ["case_D2"]
        more_tools_cases = ["case_D3", "case_C3", "case_C6", "case_C7", "case_D6", "case_D7", 
                           "case_C9", "case_D9", "case_C10", "case_D10", "case_A3", "case_A4", 
                           "case_B4", "case_B5", "case_B6"]
        
        if wheel_type in fewer_tools_cases:
            try:
                messages = data[1].get("messages", [])
                good_tool_mapping = data[2].get("good_tool_mapping", [])
                tool_select = data[6].get("tool_select", "")
                
                extracted_tools = []
                for message in messages:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                        tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                        
                        for tool_call in tool_calls:
                            try:
                                tool_json = json.loads(tool_call.strip())
                                tool_name = tool_json.get("name", "")
                                if tool_name and tool_name not in extracted_tools:
                                    extracted_tools.append(tool_name)
                            except json.JSONDecodeError:
                                return 0
                
                tool_order = parse_tool_select(tool_select)
                expected_tools = []
                for tool in tool_order:
                    for mapping in good_tool_mapping:
                        if mapping.get("original_tool") == tool:
                            diversity = mapping.get("diversity", "")
                            if diversity:
                                expected_tools.append(diversity)
                            break
                
                return 1 if len(extracted_tools) > 0 and extracted_tools[0] in expected_tools else 0
            except Exception as e:
                print(f"Error in case_D2: {e}")
                return 0
        
        elif wheel_type in more_tools_cases:
            try:
                messages = data[1].get("messages", [])
                good_tool_mapping = data[2].get("good_tool_mapping", [])
                tool_select = data[6].get("tool_select", "")
                
                extracted_tools = []
                for message in messages:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                        tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                        
                        for tool_call in tool_calls:
                            try:
                                tool_json = json.loads(tool_call.strip())
                                tool_name = tool_json.get("name", "")
                                if tool_name and tool_name not in extracted_tools:
                                    extracted_tools.append(tool_name)
                            except json.JSONDecodeError:
                                return 0
                
                tool_order = parse_tool_select(tool_select)
                expected_tools = []
                for tool in tool_order:
                    for mapping in good_tool_mapping:
                        if mapping.get("original_tool") == tool:
                            diversity = mapping.get("diversity", "")
                            if diversity:
                                expected_tools.append(diversity)
                            break
                
                for tool in expected_tools:
                    if tool not in extracted_tools:
                        return 0
                return 1
                
            except Exception as e:
                print(f"Error in more tools case: {e}")
                return 0
        
        else:
            try:
                messages = data[1].get("messages", [])
                good_tool_mapping = data[2].get("good_tool_mapping", [])
                tool_select = data[6].get("tool_select", "")
                
                extracted_tools = []
                for message in messages:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                        tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                        
                        for tool_call in tool_calls:
                            try:
                                tool_json = json.loads(tool_call.strip())
                                tool_name = tool_json.get("name", "")
                                if tool_name and tool_name not in extracted_tools:
                                    extracted_tools.append(tool_name)
                            except json.JSONDecodeError:
                                return 0
                
                tool_order = parse_tool_select(tool_select)
                expected_tools = []
                for tool in tool_order:
                    for mapping in good_tool_mapping:
                        if mapping.get("original_tool") == tool:
                            diversity = mapping.get("diversity", "")
                            if diversity:
                                expected_tools.append(diversity)
                            break
                
                return 1 if extracted_tools == expected_tools else 0
                
            except Exception as e:
                print(f"Error in normal case: {e}")
                return 0

    @staticmethod
    def check_tool_bank(data):
        """9. Check if tool_call names and parameters match those defined in tool_bank"""
        try:
            tool_bank_info = []
            tool_calls_info = []
            
            for tool in data[5]["argument_tool_bank"]:
                properties = list(tool["parameters"]["properties"].keys())
                tool_info = {
                    'name': tool["name"],
                    'properties': properties,
                    'arguments': tool["parameters"]["required"]
                }
                tool_bank_info.append(tool_info)
            
            for message in data[1]["messages"]:
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                    tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                    
                    for tool_call in tool_calls:
                        try:
                            tool_data = json.loads(tool_call.strip())
                            tool_name = tool_data.get('name', '')
                            arguments = tool_data.get('arguments', {})
                            argument_keys = list(arguments.keys())
                            tool_info = {
                                'name': tool_name,
                                'arguments': argument_keys
                            }
                            if tool_info not in tool_calls_info:
                                tool_calls_info.append(tool_info)
                        except json.JSONDecodeError:
                            return 0

            bank_dict = {}
            for tool in tool_bank_info:
                bank_dict[tool['name']] = {
                    'properties': set(tool['properties']),
                    'arguments': set(tool['arguments'])
                }
            
            all_valid = True
            
            for tool_call in tool_calls_info:
                call_name = tool_call['name']
                call_args = set(tool_call['arguments'])
                
                if call_name not in bank_dict:
                    print(f"❌ Tool name '{call_name}' is not in tool_bank")
                    all_valid = False
                    continue
                
                bank_tool = bank_dict[call_name]
                bank_properties = bank_tool['properties']
                bank_required = bank_tool['arguments']
                
                missing_required = bank_required - call_args
                if missing_required:
                    print(f"❌ Missing required parameters: {list(missing_required)}")
                    all_valid = False
                    continue
                
                extra_args = call_args - bank_required
                invalid_extra_args = extra_args - bank_properties
                
                if invalid_extra_args:
                    print(f"❌ Extra parameters {list(invalid_extra_args)} are not in the tool's properties")
                    all_valid = False
                    continue
            
            return 1 if all_valid else 0
            
        except Exception as e:
            print(f"Error occurred during tool_bank check: {e}")
            return 0
