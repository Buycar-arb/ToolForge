"""
Feature 3: Data Generation and Validation
Integrates stage_2_generate and stage_3_judge functionality
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from collections import deque
from pathlib import Path

# Add project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import generation and validation related modules
from stage_2_generate.generate_and_judge_main import IntegratedDataGenerator, process_and_validate_single_data

class FeatureGenerateJudgeProcessor:
    """Data generation and validation processor"""
    
    def __init__(self):
        self.is_running = False
        self.current_input_file = ""
        self.current_output_files = {}  # {case_name: {"data": path, "score": path}}
        self.process_logs = deque(maxlen=200)  # Increased log capacity
        self.data_generator = None
    
    def process_data_generate_judge(
        self,
        input_file: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt_file: str,
        user_prompt_file: str,
        tool_bank_file: str,
        virtual_tool_min: int,
        virtual_tool_max: int,
        cases_config_json: str,  # JSON format case configuration
        progress=None
    ):
        """
        Main data generation and validation processing function
        
        Args:
            input_file: Input file path
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum token count
            system_prompt_file: System prompt file
            user_prompt_file: User prompt file
            tool_bank_file: Tool bank file
            virtual_tool_min: Minimum virtual tool count
            virtual_tool_max: Maximum virtual tool count
            cases_config_json: Case configuration (JSON format)
            progress: Gradio progress object
        
        Returns:
            tuple: (status message, log message)
        """
        try:
            self.is_running = True
            self.process_logs.clear()
            self.current_input_file = input_file
            
            # Parse cases configuration
            try:
                cases_config = json.loads(cases_config_json)
            except json.JSONDecodeError as e:
                return f"❌ Cases configuration JSON parsing failed: {str(e)}", ""
            
            # Validate input
            if progress:
                progress(0, desc="🔍 Validating input...")
            
            if not os.path.exists(input_file):
                return f"❌ Input file does not exist: {input_file}", ""
            
            # Run async processing
            result_msg, log_msg = asyncio.run(self._async_process(
                input_file, model, temperature, max_tokens,
                system_prompt_file, user_prompt_file, tool_bank_file,
                virtual_tool_min, virtual_tool_max, cases_config, progress
            ))
            
            return result_msg, log_msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Processing failed: {str(e)}\n{traceback.format_exc()}"
            return error_msg, self._generate_log()
        finally:
            self.is_running = False
    
    async def _async_process(
        self, input_file, model, temperature, max_tokens,
        system_prompt_file, user_prompt_file, tool_bank_file,
        virtual_tool_min, virtual_tool_max, cases_config, progress
    ):
        """Async processing logic"""
        import aiohttp
        
        try:
            # Initialize generator
            if progress:
                progress(0.05, desc="🔧 Initializing generator...")
            
            self.data_generator = IntegratedDataGenerator(mcp_api_url=None, model=model)
            await self.data_generator.connect_to_mcp()
            
            self._log("✅ Generator initialization completed")
            
            # Basic configuration
            base_config = {
                'system_prompt_file': system_prompt_file,
                'user_prompt_file': user_prompt_file,
                'tool_bank_file': tool_bank_file,
                'virtual_tool_number_min': virtual_tool_min,
                'virtual_tool_number_max': virtual_tool_max,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'input_file': input_file,
            }
            
            # Create output directories
            if progress:
                progress(0.1, desc="📁 Creating output directories...")
            
            self.current_output_files = {}
            for case_name, case_info in cases_config.items():
                data_path = case_info['data_output']
                score_path = case_info['score_output']
                
                # Create directories and files
                for file_path in [data_path, score_path]:
                    file_obj = Path(file_path)
                    file_obj.parent.mkdir(parents=True, exist_ok=True)
                    if not file_obj.exists():
                        file_obj.touch()
                
                self.current_output_files[case_name] = {
                    "data": data_path,
                    "score": score_path
                }
                
                self._log(f"✅ Created output file: {case_name}")
            
            # Read input data
            if progress:
                progress(0.15, desc="📚 Reading input data...")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            self._log(f"📚 Read {total_lines} original data items")
            
            # Statistics counters
            case_success_counts = {case: 0 for case in cases_config.keys()}
            case_attempt_counts = {case: 0 for case in cases_config.keys()}
            
            # Process each case
            async with aiohttp.ClientSession() as session:
                total_cases = len(cases_config)
                current_case_idx = 0
                
                for case_name, case_info in cases_config.items():
                    current_case_idx += 1
                    target_count = case_info['target_count']
                    data_output = case_info['data_output']
                    score_output = case_info['score_output']
                    
                    self._log(f"\n{'='*50}")
                    self._log(f"Starting processing {case_name}, target count: {target_count}")
                    self._log(f"{'='*50}")
                    
                    line_index = 0
                    max_attempts = total_lines * 2
                    
                    while (case_success_counts[case_name] < target_count and 
                           case_attempt_counts[case_name] < max_attempts):
                        
                        if line_index >= total_lines:
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
                            
                            # Update progress
                            if progress:
                                base_progress = 0.15 + (current_case_idx - 1) / total_cases * 0.7
                                case_progress = (case_success_counts[case_name] / target_count) * (0.7 / total_cases)
                                total_progress = base_progress + case_progress
                                progress(
                                    total_progress,
                                    desc=f"🔄 Processing {case_name}: {case_success_counts[case_name]}/{target_count}"
                                )
                            
                            # Process and validate data
                            success, message = await process_and_validate_single_data(
                                self.data_generator, data_item, base_config,
                                data_output, score_output,
                                case_attempt_counts[case_name], case_name, session
                            )
                            
                            if success:
                                case_success_counts[case_name] += 1
                                self._log(f"✅ {case_name} progress: {case_success_counts[case_name]}/{target_count}")
                            else:
                                self._log(f"❌ Attempt {case_attempt_counts[case_name]} failed: {message}")
                            
                            # Delay to avoid API rate limiting
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            self._log(f"❌ Data processing exception: {e}")
                            continue
                    
                    # Case completion summary
                    success_rate = (case_success_counts[case_name] / case_attempt_counts[case_name] * 100) if case_attempt_counts[case_name] > 0 else 0
                    self._log(f"\n{case_name} completed!")
                    self._log(f"Success: {case_success_counts[case_name]}/{target_count}")
                    self._log(f"Total attempts: {case_attempt_counts[case_name]}")
                    self._log(f"Success rate: {success_rate:.2f}%")
            
            if progress:
                progress(0.9, desc="📊 Generating statistics report...")
            
            # Final statistics
            total_success = sum(case_success_counts.values())
            total_attempts = sum(case_attempt_counts.values())
            overall_success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
            
            result_msg = f"""
✅ Processing completed!

📊 Overall statistics:
   - Successfully generated data: {total_success} items
   - Total attempts: {total_attempts} times
   - Overall success rate: {overall_success_rate:.2f}%

📋 Case details:
"""
            
            for case_name, case_info in cases_config.items():
                target = case_info['target_count']
                success = case_success_counts[case_name]
                attempts = case_attempt_counts[case_name]
                success_rate = (success / attempts * 100) if attempts > 0 else 0
                status = "✅" if success >= target else "❌"
                
                result_msg += f"\n{status} {case_name}: {success}/{target} (Success rate: {success_rate:.2f}%)"
                result_msg += f"\n   - Data file: {case_info['data_output']}"
                result_msg += f"\n   - Score file: {case_info['score_output']}"
            
            result_msg += f"\n\n⏰ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            if progress:
                progress(1.0, desc="✅ All completed!")
            
            # Clean up resources
            await self.data_generator.close()
            
            return result_msg, self._generate_log()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Async processing failed: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg)
            
            if self.data_generator:
                await self.data_generator.close()
            
            return error_msg, self._generate_log()
    
    def _log(self, message):
        """Record log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        self.process_logs.append(log_entry)
        print(message)  # Also output to console
    
    def _generate_log(self):
        """Generate log text"""
        if not self.process_logs:
            return "No processing logs available"
        
        log_text = f"📋 Processing logs (total {len(self.process_logs)} items)\n\n"
        log_text += "="*80 + "\n\n"
        
        for log in self.process_logs:
            log_text += f"{log}\n"
        
        return log_text
    
    # ========== File operation methods ==========
    
    def load_jsonl_file(self, file_path):
        """Load JSONL file"""
        if not file_path or not os.path.exists(file_path):
            return []
        
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        data.append((i, json.loads(line)))
                    except json.JSONDecodeError:
                        data.append((i, {"error": "JSON parsing failed"}))
        except Exception as e:
            return [(0, {"error": f"File reading failed: {str(e)}"})]
        
        return data
    
    def get_line_content(self, file_path, line_number):
        """Get content of specified line"""
        data = self.load_jsonl_file(file_path)
        
        if not data:
            return "File is empty or does not exist"
        
        if line_number < 1 or line_number > len(data):
            return f"Line number out of range (1-{len(data)})"
        
        line_num, content = data[line_number - 1]
        return json.dumps(content, ensure_ascii=False, indent=2)
    
    def get_file_info(self, file_path):
        """Get file information"""
        if not file_path or not os.path.exists(file_path):
            return "File does not exist", 0
        
        data = self.load_jsonl_file(file_path)
        total_lines = len(data)
        
        info = f"""
📄 File path: {file_path}
📊 Total lines: {total_lines}
📅 Modified time: {datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')}
💾 File size: {os.path.getsize(file_path) / 1024:.2f} KB
"""
        return info, total_lines
    
    def get_all_output_files(self):
        """Get all output file list"""
        files = []
        if self.current_input_file:
            files.append(("Input file", self.current_input_file))
        
        for case_name, paths in self.current_output_files.items():
            files.append((f"{case_name}-Data", paths["data"]))
            files.append((f"{case_name}-Score", paths["score"]))
        
        return files


# ==================== UI Integration Code Example (for reference) ====================
"""
Add the following code in quick_fast.py:

# 1. Import
from feature_generate_judge import FeatureGenerateJudgeProcessor

# 2. Create instance
generator_gen_judge = FeatureGenerateJudgeProcessor()

# 3. Add Tab in UI (see next file)
"""

