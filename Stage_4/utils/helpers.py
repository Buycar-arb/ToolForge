"""Utility Functions"""

import json
from typing import Dict, Any


def parse_llm_result(result: str) -> tuple:
    """Parse the result returned by the LLM"""
    try:
        temp = result.split("<score>\n")
        score = temp[1].split("\n</score>")
        gpt_score = int(score[0][1:-1])
        return gpt_score, None
    except Exception as e:
        return "null", f"Failed to parse LLM result: {str(e)}"


def build_output_data(data: Any, case: str, rule_score: int, gpt_score: Any, 
                     total_score: int, result: str = None, failure_reason: str = None) -> Dict:
    """Build the output data dictionary"""
    output_data = {
        "data": data[1],
        "case": case,
        "rule_score": rule_score,
        "gpt_score": gpt_score,
        "total_score": total_score,
    }
    
    if total_score == 2:
        output_data["good_reason"] = result
    else:
        output_data["error_reason"] = failure_reason
    
    return output_data


def save_output_data(f_out, output_data: Dict):
    """Save the output data to file"""
    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
    f_out.flush()
