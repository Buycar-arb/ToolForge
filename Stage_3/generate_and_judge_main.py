import asyncio
import json
from pathlib import Path
from collections import defaultdict
import aiohttp
import os
import uuid
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# === Generation-related modules ===
from stage_2_generate.core.api_client import APICaller
from stage_2_generate.core.mcp_client import MCPCaller
from stage_2_generate.services.data_processor import DataProcessor
from stage_2_generate.services.tool_manager import ToolManager
from stage_2_generate.services.conversation_generator import ConversationGenerator
from stage_2_generate.config.settings import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

# === Validation-related modules ===
from stage_3_judge.config.api_keys import API_KEYS
from stage_3_judge.config.settings import API_CONFIG
from stage_3_judge.core.llm_client import AsyncLLMGenerateLabel
from stage_3_judge.validators.validation_engine import ValidationEngine
from stage_3_judge.utils.helpers import parse_llm_result
from stage_3_judge.prompts.end_judge_prompts import system_prompt, user_prompt

# Cases that require fallback checks
CASES_NEED_GENERAL_CHECK = {"case_A4", "case_B6", "case_C9", "case_C10", "case_D9", "case_D10"}

class IntegratedDataGenerator:
    def __init__(self, mcp_api_url: str = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.mcp_api_url = mcp_api_url
        
        # Initialize generation components
        self.api_caller = APICaller(model=model)
        self.mcp_client = MCPCaller() if mcp_api_url else None
        self.data_processor = DataProcessor()
        self.tool_manager = ToolManager()
        
        # Initialize validation components
        self.llm_generator = AsyncLLMGenerateLabel(
            API_KEYS, 
            API_CONFIG["url"], 
            API_CONFIG["model"], 
            API_CONFIG["temperature"], 
            API_CONFIG["max_tokens"]
        )
        self.validation_engine = ValidationEngine()
    
    async def connect_to_mcp(self):
        """Connect to the MCP server"""
        if not self.mcp_client or not self.mcp_api_url:
            return
        print("Connecting to MCP server...")
        try:
            await self.mcp_client.connect_to_server(self.mcp_api_url)
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            self.mcp_client = None

    async def call_claude_api(self, session, model, temperature, max_tokens, messages, system_prompt):
        """
        Call the LLM API (supports both Claude and GPT models)
        
        The method automatically detects the model type based on the model name:
        - GPT models (gpt-3.5, gpt-4, gpt-4-turbo, gpt-4o, etc.)
        - Claude models (claude-3, claude-sonnet, etc.)
        
        Args:
            session: aiohttp session (not used, kept for compatibility)
            model: model name (used for detection)
            temperature: temperature parameter
            max_tokens: max tokens to generate
            messages: list of message dicts
            system_prompt: system prompt to prepend
        
        Returns:
            dict with 'content' key containing the response text
        """
        if system_prompt:
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages_with_system = messages
            
        response_txt = await self.api_caller.generate(messages_with_system, max_tokens, temperature)
        
        if response_txt is None:
            return {"content": ""}
            
        return {"content": response_txt}

    def load_multihop_data_from_jsonl(self, message):
        """Load multi-hop reasoning data"""
        return self.data_processor.load_multihop_data_from_jsonl(message)

    def load_prompts(self, system_prompt_file, user_prompt_file, tool_bank_file, good_tool, vritual_tool_number_min, vritual_tool_number_max):
        """Load prompts"""
        return self.tool_manager.load_prompts(
            system_prompt_file, user_prompt_file, tool_bank_file, 
            good_tool, vritual_tool_number_min, vritual_tool_number_max
        )
    
    async def generate_multihop_data(self, cases, **kwargs):
        """Generate multi-hop data"""
        kwargs['call_claude_api'] = self.call_claude_api
        kwargs['call_llm_api'] = None
        
        generator = ConversationGenerator(**kwargs)
        return await generator.process(cases)
    
    async def validate_generated_data(self, complete_data, original_data, case_type, session):
        """Validate generated data and return scoring information"""
        try:
            print("1. Start constructing the validation data format...")
            
            if len(complete_data) < 3:
                return False, "Incomplete generated data format", None
            
            # Construct the validation data structure
            validation_data = [
                complete_data[0],  # case info
                complete_data[1],  # messages
                complete_data[2],  # metadata
                complete_data[3] if len(complete_data) > 3 else {"argument_check": "Don't need to check"},
                complete_data[4] if len(complete_data) > 4 else {},
                complete_data[5] if len(complete_data) > 5 else {"argument_tool_bank": []},
                complete_data[-1] if len(complete_data) > 6 else original_data
            ]
            
            print("2. Start rule-based validation...")
            # Execute rule validation
            check_results, failed_checks = self.validation_engine.validate_all(validation_data, case_type)
            
            # Initialize scoring info
            rule_score = 0
            gpt_score = "null"
            total_score = 0
            failure_reason = ""
            llm_result = None
            
            # Check rule-based results
            if not all(result == 1 for result in check_results.values()):
                rule_score = 0
                failure_reason = "; ".join(failed_checks)
                total_score = rule_score
                print(f"Rule validation failed: {failure_reason}")
                
                score_info = {
                    "case": case_type,
                    "rule_score": rule_score,
                    "gpt_score": gpt_score,
                    "total_score": total_score,
                    "error_reason": failure_reason
                }
                
                return False, failure_reason, score_info
            
            # Rule validation passed
            rule_score = 1
            print("3. Rule validation passed, starting LLM validation...")
            
            # Perform LLM validation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(
                    messages=validation_data[1], 
                    good_tool_mapping=validation_data[2]["good_tool_mapping"]
                )}
            ]
            
            llm_result = await self.llm_generator.call_llm_api(session, messages)
            
            if llm_result:
                gpt_score, parse_error = parse_llm_result(llm_result)
                if parse_error:
                    failure_reason = parse_error
                    gpt_score = "null"
                    total_score = rule_score
                elif gpt_score == 0:
                    failure_reason = llm_result
                    total_score = rule_score + gpt_score
                else:
                    print(f"LLM validation passed, score: {gpt_score}")
                    total_score = rule_score + gpt_score
            else:
                failure_reason = "LLM call failed"
                gpt_score = "null"
                total_score = rule_score
            
            # Construct score information
            score_info = {
                "case": case_type,
                "rule_score": rule_score,
                "gpt_score": gpt_score,
                "total_score": total_score
            }
            
            # Add reason based on total score
            if total_score == 2:
                score_info["good_reason"] = llm_result
            else:
                score_info["error_reason"] = failure_reason
            
            # Determine whether validation passed
            is_valid = (total_score == 2)
            validation_message = "Validation passed" if is_valid else failure_reason
            
            return is_valid, validation_message, score_info
                
        except Exception as e:
            print(f"Exception during validation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            score_info = {
                "case": case_type,
                "rule_score": 0,
                "gpt_score": "null",
                "total_score": 0,
                "error_reason": f"Validation exception: {str(e)}"
            }
            
            return False, f"Validation exception: {str(e)}", score_info
    
    async def close(self):
        """Close resources"""
        try:
            if self.mcp_client:
                await self.mcp_client.cleanup()
            print("Resources cleaned up successfully")
        except Exception as e:
            print(f"Error during resource cleanup: {e}")


def save_validated_result(result, output_file):
    """Save validated results to file"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
        return True
    except Exception as e:
        print(f"Error saving result: {e}")
        return False


def save_score_result(score_info, score_output_file):
    """Save scoring results to file"""
    try:
        with open(score_output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(score_info, ensure_ascii=False) + '\n')
            f.flush()
        return True
    except Exception as e:
        print(f"Error saving score result: {e}")
        return False


async def process_and_validate_single_data(data_generator, data_item, config, output_file, score_output_file, line_num, case_type, session):
    """Process and validate a single data item"""
    try:
        print(f"\n=== Start processing line {line_num}, target case: {case_type} ===")
        
        # Step 1: Data preprocessing
        print("Step 1: Process raw data and perform preprocessing")
        query, gold_contents, all_contents, answer, original_case_type, reasoning = data_generator.load_multihop_data_from_jsonl(data_item)
        
        good_tool = data_item["tool_select"]
        good_tool = good_tool.strip('[]').replace(' ', '').split(',')
        
        (random_all_tools, system_prompt, tool_prompt, good_tool_content, 
         user_prompt, general_tool_name_and_content, tool_prompt_general, random_all_tools_general, 
         good_tool_mapping, simulate_recall_tools_json, simulate_recall_tools_general_json) = data_generator.load_prompts(
            config['system_prompt_file'],
            config['user_prompt_file'], 
            config['tool_bank_file'],
            good_tool,
            config['virtual_tool_number_min'],
            config['virtual_tool_number_max']
        )
        
        if case_type in CASES_NEED_GENERAL_CHECK and tool_prompt_general == "error":
            print("During fallback construction, the tool_select includes the fallback tool, skipping this item.")
            return False, "Fallback tool check failed"
        
        print(f"Step 1 completed, moving to Step 2: generate conversation data (main type: {original_case_type}, sub type: {case_type})")
        result = await data_generator.generate_multihop_data(
            cases=case_type,
            system_prompt=system_prompt,
            random_all_tools_content=random_all_tools,
            tool_prompt=tool_prompt,
            good_tool_content=good_tool_content,
            user_prompt=user_prompt,
            general_tool_name=general_tool_name_and_content,
            tool_prompt_general=tool_prompt_general,
            random_all_tools_content_general=random_all_tools_general,
            good_tool_mapping=good_tool_mapping,
            simulate_recall_tools_json=simulate_recall_tools_json,
            simulate_recall_tools_general_json=simulate_recall_tools_general_json,
            query=query,
            gold_contents=gold_contents,
            all_contents=all_contents,
            answer=answer,
            max_tokens=config.get('max_tokens', DEFAULT_MAX_TOKENS),
            temperature=config.get('temperature', DEFAULT_TEMPERATURE),
            wheel_type=original_case_type,
            reasoning=reasoning
        )

        if result is None:
            print(f"Line {line_num} generation result is None, skipping")
            return False, "Generation result is None"
        
        # Construct full data for validation
        unique_id = str(uuid.uuid4())
        complete_data = [
            dict(case=case_type, uuid=unique_id),
            *result, 
            data_item
        ]
        
        # Step 2: Validate data and get scores
        print("Step 2: Start data validation")
        is_valid, validation_message, score_info = await data_generator.validate_generated_data(
            complete_data, data_item, case_type, session
        )
        
        # Add UUID and generated data to score info
        if score_info:
            score_info["uuid"] = unique_id
            score_info["data"] = complete_data[1]
        
        # Step 3: Save score info (always save)
        if score_info:
            save_score_result(score_info, score_output_file)
            print(f"Score saved: rule={score_info['rule_score']}, GPT={score_info['gpt_score']}, total={score_info['total_score']}")
        
        if not is_valid:
            print(f"Line {line_num} validation failed: {validation_message}")
            return False, validation_message
        
        # Step 4: Save successfully validated data
        print("Step 4: Save validated data")
        success = save_validated_result(complete_data, output_file)
        
        if success:
            print(f"✅ Line {line_num} processed successfully and saved")
            return True, "Success"
        else:
            print(f"❌ Line {line_num} save failed")
            return False, "Save failed"
        
    except Exception as e:
        print(f"❌ Error processing line {line_num}: {e}")
        import traceback
        traceback.print_exc()
        
        error_score_info = {
            "case": case_type,
            "rule_score": 0,
            "gpt_score": "null", 
            "total_score": 0,
            "error_reason": f"Processing exception: {str(e)}",
            "uuid": str(uuid.uuid4()) if 'unique_id' not in locals() else unique_id,
            "data": result[0] if 'result' in locals() and result else None
        }
        save_score_result(error_score_info, score_output_file)
        
        return False, f"Processing exception: {str(e)}"


async def main():
    """Main function"""
    # Configuration for case C: case_name -> (limit, validated_output_file, score_output_file)
    c_cases_config = {
        'case_C1': (
            1, 
            "/.../validated_case_C1.jsonl",
            "/.../score_case_C1.jsonl"
        ),
        # Additional cases can be added...
    }
    
    base_config = {
        'system_prompt_file': os.path.join(BASE_DIR, "stage_2_generate", "prompts", "opt_sys_prompt.txt"),
        'user_prompt_file': os.path.join(BASE_DIR, "stage_2_generate", "prompts", "opt_user_prompt.txt"),
        'tool_bank_file': os.path.join(BASE_DIR, "stage_2_generate", "tool_bank", "tools"),
        'virtual_tool_number_min': 3,
        'virtual_tool_number_max': 8,
        'max_tokens': 8192,
        'temperature': 0.0,
        'input_file': "Your Preprocess ==> xxx_judge.jsonl",
    }
    
    # Initialize integrated data generator
    data_generator = IntegratedDataGenerator(mcp_api_url=None, model=DEFAULT_MODEL)
    
    try:
        await data_generator.connect_to_mcp()
        
        # Read input data
        input_file = Path(base_config['input_file'])
        if not input_file.exists():
            print(f"Input file does not exist: {input_file}")
            return
        
        # Create output directories
        for case_name, (limit, output_path, score_path) in c_cases_config.items():
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if not output_file.exists():
                output_file.touch()
                print(f"Created validated data file for {case_name}: {output_file}")
            
            score_file = Path(score_path)
            score_file.parent.mkdir(parents=True, exist_ok=True)
            if not score_file.exists():
                score_file.touch()
                print(f"Created score file for {case_name}: {score_file}")
        
        # Count success attempts per case
        case_success_counts = {case: 0 for case in c_cases_config.keys()}
        case_attempt_counts = {case: 0 for case in c_cases_config.keys()}
        
        # Read all raw data
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total {len(lines)} raw data items read")
        
        # Generate and validate data for each case
        async with aiohttp.ClientSession() as session:
            for case_name, (target_count, output_path, score_path) in c_cases_config.items():
                print(f"\n{'='*50}")
                print(f"Start generating data for {case_name}, target count: {target_count}")
                print(f"{'='*50}")
                
                line_index = 0
                max_attempts_per_case = len(lines) * 2
                
                while (case_success_counts[case_name] < target_count and 
                       case_attempt_counts[case_name] < max_attempts_per_case):
                    
                    if line_index >= len(lines):
                        line_index = 0
                    
                    line = lines[line_index].strip()
                    line_index += 1
                    case_attempt_counts[case_name] += 1
                    
                    if not line:
                        continue
                    
                    try:
                        data_item = json.loads(line)
                        
                        if 'tool_select' not in data_item:
                            continue
                        
                        # Process and validate data
                        success, message = await process_and_validate_single_data(
                            data_generator, data_item, base_config, output_path, score_path,
                            case_attempt_counts[case_name], case_name, session
                        )
                        
                        if success:
                            case_success_counts[case_name] += 1
                            print(f"🎉 {case_name} progress: {case_success_counts[case_name]}/{target_count} (attempts: {case_attempt_counts[case_name]})")
                        else:
                            print(f"❌ Attempt {case_attempt_counts[case_name]} failed: {message}")
                        
                        # Add delay to avoid API limits
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        print(f"Data processing exception: {e}")
                        continue
                
                success_rate = (case_success_counts[case_name] / case_attempt_counts[case_name] * 100) if case_attempt_counts[case_name] > 0 else 0
                print(f"\n{case_name} completed!")
                print(f"Validated successfully: {case_success_counts[case_name]}/{target_count}")
                print(f"Total attempts: {case_attempt_counts[case_name]}")
                print(f"Success rate: {success_rate:.2f}%")
        
        # Final summary
        print(f"\n{'='*50}")
        print("Final Summary")
        print(f"{'='*50}")
        total_success = 0
        total_attempts = 0
        
        for case_name in c_cases_config.keys():
            target = c_cases_config[case_name][0]
            success = case_success_counts[case_name]
            attempts = case_attempt_counts[case_name]
            success_rate = (success / attempts * 100) if attempts > 0 else 0
            
            total_success += success
            total_attempts += attempts
            
            status = "✅" if success >= target else "❌"
            print(f"{status} {case_name}: {success}/{target} (Success rate: {success_rate:.2f}%)")
        
        overall_success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
        print(f"\nTotal:")
        print(f"Validated successfully generated data: {total_success}")
        print(f"Total attempts: {total_attempts}")
        print(f"Overall success rate: {overall_success_rate:.2f}%")
        
    except Exception as e:
        print(f"Main program execution error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await data_generator.close()

if __name__ == "__main__":
    asyncio.run(main())
