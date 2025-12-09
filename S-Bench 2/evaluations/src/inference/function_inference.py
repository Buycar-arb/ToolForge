"""Function-based inference implementation."""

import json
from typing import Dict, Any
from ..search.function_search import FunctionSearchHandler


class FunctionInference:
    """Handle function-based inference with multiple tool support."""

    def __init__(self, model, search_handler: FunctionSearchHandler, prompt_config: Dict):
        """
        Initialize function inference.

        Args:
            model: Language model instance
            search_handler: Function search handler
            prompt_config: Prompt configuration
        """
        self.model = model
        self.search_handler = search_handler
        self.prompt_config = prompt_config
        self.max_iterations = 10

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run inference with function tools.

        Args:
            question: Input question

        Returns:
            Dictionary containing answer and messages
        """
        tools = self.search_handler.get_tool_schemas()

        # Initialize message history
        messages = []

        is_open_source = 'open_source' in str(self.prompt_config).lower() or '{{TOOLS_PLACEHOLDER}}' in self.prompt_config.get('system', '')

        if self.prompt_config.get('system'):
            system_content = self.prompt_config['system']

            if is_open_source and '{{TOOLS_PLACEHOLDER}}' in system_content:
                tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
                system_content = system_content.replace('{{TOOLS_PLACEHOLDER}}', tools_json)

            messages.append({
                "role": "system",
                "content": system_content
            })

        messages.append({
            "role": "user",
            "content": self.prompt_config['user'].format(question=question)
        })

        iterations = 0
        final_answer = None

        try:
            while iterations < self.max_iterations:
                iterations += 1

                response = self.model.generate_with_functions(messages, tools)
                function_calls = self.search_handler.parse_tool_calls(response) # check if <tool_call>...</tool_call> exists in the response

                if function_calls:
                    # Handle assistant message based on model type
                    if 'tool_calls' in response and response['tool_calls']:
                        # Closed source models (OpenAI, DeepSeek) - preserve tool_calls field
                        messages.append({
                            "role": "assistant",
                            "content": response.get('content', ''),
                            "tool_calls": response['tool_calls']
                        })
                    else:
                        # Open source models with XML format - content only
                        messages.append({
                            "role": "assistant",
                            "content": response.get('content', '')
                        })

                    # Execute each function call
                    for call in function_calls:
                        function_name = call['name']
                        arguments = call['arguments']

                        result = self.search_handler.call_function(function_name, arguments)

                        tool_response = self.search_handler.format_tool_response(
                            call['id'],
                            result
                        )
                        messages.append(tool_response)

                else:
                    # No function calls, check for final answer
                    if response.get('content'):
                        # Add final assistant message
                        messages.append({
                            "role": "assistant",
                            "content": response['content']
                        })

                        # Try to extract answer
                        final_answer = self.search_handler.extract_final_answer(response['content'])

                        # If we have an answer, return results
                        if final_answer:
                            return {
                                'answer': final_answer,
                                'messages': messages
                            }

                        # If no answer found but content exists, continue if we haven't reached max iterations
                        if iterations >= self.max_iterations:
                            # Use the last content as answer if no explicit answer tags
                            final_answer = response['content']
                            break
                    else:
                        # No content and no function calls - something went wrong
                        break

            return {
                'answer': final_answer,
                'messages': messages
            }

        except Exception as e:
            print(e)
            return {
                'answer': None,
                'error': str(e),
                'messages': messages  # Simplified: only keep messages
            }

