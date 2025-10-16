"""Validation Engine"""

from typing import Tuple, Dict, List
from .base_validator import DataValidator
from stage_3_judge.config.settings import DIALOGUE_PATTERNS, ARGUMENT_CHECK_CONFIG, ERROR_MESSAGES  # Used during generation + validation
# from config.settings import DIALOGUE_PATTERNS, ARGUMENT_CHECK_CONFIG, ERROR_MESSAGES  # Used for standalone validation only

class ValidationEngine:
    """Validation Engine Class"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.check_functions = [
            ("format_result", self._check_format),
            ("content_result", self._check_content),
            ("not_empty_result", self._check_not_empty),
            ("answer_consistency_result", self._check_answer_consistency),
            ("tool_rags_consistency_result", self._check_tool_rags_consistency),
            ("argument_check_result", self._check_arguments),
            ("reference_check_result", self._check_reference),
            ("tool_consistency_check_result", self._check_tool_consistency),
            ("tool_bank_result", self._check_tool_bank),  
        ]
    
    def _check_format(self, data, case):
        # print(DIALOGUE_PATTERNS[case])
        return self.validator.check_dialogue_format(data[1], DIALOGUE_PATTERNS[case])
    
    def _check_content(self, data, case):
        return self.validator.check_assistant_content_format(data[1])
    
    def _check_not_empty(self, data, case):
        return self.validator.check_non_assistant_content_not_empty(data[1])
    
    def _check_answer_consistency(self, data, case):
        return self.validator.check_last_assistant_answer_consistency(data)
    
    def _check_tool_rags_consistency(self, data, case):
        return self.validator.check_tool_rags_consistency(data)
    
    def _check_arguments(self, data, case):
        if data[3]["argument_check"] == "Don't need to check":
            return 1
        
        # Look up the matching parameter configuration
        for case_set, params in ARGUMENT_CHECK_CONFIG.items():
            if case in case_set:
                return self.validator.check_argument_modifications(data, *params)
        
        return 1
    
    def _check_reference(self, data, case):
        return self.validator.check_reference_consistency(data)
    
    def _check_tool_consistency(self, data, case):
        return self.validator.check_tool_consistency(data, case)
    
    def _check_tool_bank(self, data, case):
        return self.validator.check_tool_bank(data)
    
    def validate_all(self, data, case) -> Tuple[Dict[str, int], List[str]]:
        """Run all validation checks and return the results and failure reasons"""
        results = {}
        failed_checks = []
        
        for check_name, check_func in self.check_functions:
            result = check_func(data, case)
            results[check_name] = result
            
            if result == 0:
                error_msg = ERROR_MESSAGES[check_name]
                print(error_msg)
                failed_checks.append(error_msg)
        
        return results, failed_checks
