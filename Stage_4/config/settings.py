"""Configuration file"""

# API configuration
API_CONFIG = {
    "url": 'https://aigc.sankuai.com/v1/openai/native',
    "model": "gpt-4.1",
    "temperature": 0,
    "max_tokens": 8192,
}

# File path configuration
FILE_PATHS = {
    "input_file": "The output file obtained at the generate stage",
    "output_file": "Your judge result file"
}

# Dialogue pattern configuration
DIALOGUE_PATTERNS = {
    "case_A1": ["system", "user", "assistant", "tool", "assistant"],
    "case_A2": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_A3": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_A4": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_B1": ["system", "user", "assistant", "tool", "assistant"],
    "case_B2": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_B3": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_B4": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_B5": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_B6": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C1": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C3": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C4": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C5": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C6": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C7": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C8": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C9": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_C10": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D1": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D2": ["system", "user", "assistant", "tool", "assistant"],
    "case_D3": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D4": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D5": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D6": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D7": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D8": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D9": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
    "case_D10": ["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"],
}


# Argument check configuration
ARGUMENT_CHECK_CONFIG = {
    frozenset(["case_C4", "case_D4", "case_B6", "case_A4", "case_C10", "case_D10"]): (1, 2),
    frozenset(["case_C5", "case_D5"]): (0, 3),
    frozenset(["case_C9", "case_D9"]): (2, 3),
    frozenset(["case_C8", "case_D8", "case_A2", "case_B2", "case_B3"]): (0, 1),
}


# Tool consistency check configuration
TOOL_CONSISTENCY_CONFIG = {
    "fewer_tools": {"case_D2"},
    "more_tools": {"case_D3", "case_C3", "case_C6", "case_C7", "case_D6", "case_D7", 
                   "case_C9", "case_D9", "case_C10", "case_D10", "case_A3", "case_A4", 
                   "case_B4", "case_B5", "case_B6"},
}

# Error message configuration
ERROR_MESSAGES = {
    "format_result": "1. Dialogue format validation failed",
    "content_result": "2. Assistant content format validation failed",
    "not_empty_result": "3. Non-assistant field empty validation failed",
    "answer_consistency_result": "4. Answer consistency check failed",
    "tool_rags_consistency_result": "5. Tool–RAG consistency check failed",
    "argument_check_result": "6. Argument validation failed",
    "reference_check_result": "7. Reference error at one or more stages",
    "tool_consistency_check_result": "8. Predefined tool count mismatch or inconsistent usage order",
    "tool_bank_result": "9. Mismatch between tool_call names/arguments and tool_bank definitions", 
}
