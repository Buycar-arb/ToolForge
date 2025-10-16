# Text processing utility functions
import re
import json
import ast

def extract_tags_as_str_list(text, tag, return_as_list=True):
    """
    Extract multiple content blocks enclosed by a given tag (e.g., <tool_call>...</tool_call>).
    Each extracted block is returned as a string.

    Args:
        text (str): The input text containing tagged content.
        tag (str): The tag name to extract (without angle brackets).
        return_as_list (bool): Whether to return a list (default: True).
                              If False, returns a single concatenated string.

    Returns:
        list[str] | str: List of extracted strings, or a single combined string.
    """
    # Regular expression to extract all blocks between <tag>...</tag>
    tool_call_pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    result_list = [match for match in matches]
    
    if return_as_list:
        return result_list
    else:
        # Join multiple matches with newline characters
        return '\n'.join(result_list) if result_list else ""


def parse_jsonl_string(content: str) -> list:
    """
    Parse a model output string containing a JSON block and return its 'messages' array.
    Automatically removes the first 'user' message if present.

    Args:
        content (str): The model output text containing a JSON structure (usually fenced by ```json ... ```).

    Returns:
        list[dict] | None: Parsed list of message dictionaries, or None if parsing fails.
    """
    try:
        # Extract the JSON portion between ```json and ```
        json_pattern = r'```json\s*\n(.*?)\n```'
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            parsed_json = json.loads(json_str)
            
            # Retrieve the 'messages' array
            messages = parsed_json.get('messages', [])
            
            # Remove the first 'user' message if applicable
            if messages and messages[0].get('role') == 'user':
                messages = messages[1:]
            
            return messages
        
        return None
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted content: {json_str[:200] if 'json_str' in locals() else 'N/A'}...")
        return None
    except Exception as e:
        print(f"Unexpected error during parsing: {e}")
        return None


def extract_tool_calls_as_str_list(text):
    """
    Extract multiple <tool_call>...</tool_call> blocks from text.
    Each tool call (JSON block) is returned as a string.

    Args:
        text (str): The input text containing <tool_call> tags.

    Returns:
        list[str]: A list of tool call strings.
    """
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    return [match for match in matches]


def extract_reference1_simple(text):
    """
    Extract the text content inside <reference1>...</reference1> tags.

    Args:
        text (str): The input text.

    Returns:
        str: Extracted reference text (trimmed), or an empty string if not found.
    """
    reference_pattern = r'<reference1>\s*(.*?)\s*</reference1>'
    matches = re.findall(reference_pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()
    return ""


def extract_reference2_simple(text):
    """
    Extract the text content inside <reference2>...</reference2> tags.

    Args:
        text (str): The input text.

    Returns:
        str: Extracted reference text (trimmed), or an empty string if not found.
    """
    reference_pattern = r'<reference2>\s*(.*?)\s*</reference2>'
    matches = re.findall(reference_pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()
    return ""
