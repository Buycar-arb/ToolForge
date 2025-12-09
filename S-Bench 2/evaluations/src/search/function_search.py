"""Function-based search implementation."""

import json
import re
from typing import Dict, List, Any, Optional
from .search_interface import SearchEngine


class FunctionSearchHandler:
    """Handler for function-based search in evaluation framework."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize function search handler."""
        self.config = config
        self.search_engine = SearchEngine(config)
        self.functions = self._load_functions(config)

    def _load_functions(self, config: Dict[str, Any]) -> Dict[str, Dict]:
        """Load function definitions from yaml configs."""
        functions = {}
        for func in config.get('functions', []):
            functions[func['name']] = func
        return functions

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get OpenAPI schemas for all available functions.

        Returns:
            List of function schemas in OpenAPI format
        """
        schemas = []
        for func in self.functions.values():
            schema = {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"]
                }
            }
            schemas.append(schema)
        return schemas

    def call_function(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a function and return results.

        Args:
            name: Function name to execute
            arguments: Function arguments

        Returns:
            Search results as formatted string
        """
        if name not in self.functions:
            return f"Error: Unknown function '{name}'. Available functions: {list(self.functions.keys())}"

        # Extract query and format all arguments as search query
        """
        例如，对于以下工具调用：
        arguments = {
            "query": "Einstein's birthplace and education",
            "person_identifiers": ["Albert Einstein", "Father of Relativity"],
            "info_categories": ["basic information", "education history"]
        }
        将生成搜索查询：
        query: Einstein's birthplace and education
        person_identifiers: Albert Einstein, Father of Relativity
        info_categories: basic information, education history
        """
        query_parts = []
        for param_name, param_value in arguments.items():
            if param_value is not None:
                if isinstance(param_value, list):
                    formatted_value = ', '.join(str(item) for item in param_value)
                else:
                    formatted_value = str(param_value)
                query_parts.append(f"{param_name}: {formatted_value}")

        query = '\n'.join(query_parts) if query_parts else ''
        print(query)

        try:
            results = self.search_engine.search(query)
            # results = "This is a placeholder for the search results."

            return results
        except Exception as e:
            return f"Search error: {str(e)}"


    def parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse function calls from model response.

        Args:
            response: Model response containing tool calls

        Returns:
            List of parsed function calls
        """
        tool_calls = []

        # First check if there are standard tool_calls (for closed source models)
        if 'tool_calls' in response and response['tool_calls']:
            for call in response['tool_calls']:
                parsed_call = {
                    'id': call.get('id', ''),
                    'name': call['function']['name'],
                    'arguments': json.loads(call['function']['arguments'])
                    if isinstance(call['function']['arguments'], str)
                    else call['function']['arguments']
                }
                tool_calls.append(parsed_call)

        # If no standard tool_calls, try to extract from content (for open source models with XML format)
        elif 'content' in response and '<tool_call>' in response['content']:
            import uuid

            content = response['content']
            pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                try:
                    call_data = json.loads(match.strip())
                    tool_calls.append({
                        'id': f'call_{uuid.uuid4().hex[:8]}',
                        'name': call_data['name'],
                        'arguments': call_data['arguments']
                    })
                except json.JSONDecodeError:
                    continue

        return tool_calls

    def format_tool_response(self, tool_id: str, result: str) -> Dict[str, str]:
        """
        Format tool response for model.

        Args:
            tool_id: Tool call ID
            result: Tool execution result

        Returns:
            Formatted tool response message
        """
        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result
        }

    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract final answer from model response.

        Args:
            text: Model response text

        Returns:
            Extracted answer or None if not found
        """
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return None