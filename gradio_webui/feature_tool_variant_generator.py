"""
Tool Variant Generation Feature
Based on generate_tool.py and tool_prompts.py implementation
"""

import json
import os
import asyncio
import itertools
import numpy as np
from datetime import datetime
from collections import deque
from typing import List, Dict, Any
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# Import similarity detection related modules
try:
    from sentence_transformers import SentenceTransformer
    import bm25s
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    print("⚠️ Similarity detection modules not installed, will skip similarity checking")

# Import tool prompts
try:
    import sys
    import os
    # Add project root directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from generate_virtual_tool.tool_prompts import TOOL_GENERATE_USER, TOOL_GENERATE_SYSTEM
except ImportError:
    # If unable to import, use default prompts
    TOOL_GENERATE_SYSTEM = """You are a professional tool definition rewriting expert. Please perform synonym replacement and variant generation for the given tool."""
    TOOL_GENERATE_USER = """Please generate variants for the following tool: {tool}"""

class ToolVariantGenerator:
    """Tool variant generation processor"""
    
    def __init__(self):
        # Basic state
        self.is_running = False
        self.current_input_file = ""
        self.current_output_file = ""
        
        # Generation state
        self.generation_logs = deque(maxlen=100)
        self.current_tools = []
        
        # API Configuration - Load from environment variables
        import os
        api_keys_env = os.getenv("API_KEYS", "")
        if api_keys_env:
            self.api_keys = [key.strip() for key in api_keys_env.split(",") if key.strip()]
        else:
            # Default placeholder keys
            self.api_keys = [
                "your-api-key-1",
                "your-api-key-2", 
                "your-api-key-3",
                "your-api-key-4",
                "your-api-key-5",
                "your-api-key-6",
                "your-api-key-7",
                "your-api-key-8",
                "your-api-key-9",
            ]
        
        self.api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        
        # Similarity checker
        self.similarity_checker = None
        if SIMILARITY_AVAILABLE:
            try:
                model_path = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", "./models/bge-m3")
                self.similarity_checker = AdvancedSimilarityChecker(model_path)
            except Exception as e:
                print(f"⚠️ Similarity checker initialization failed: {e}")
                self.similarity_checker = None

    def process_data_tool_variant(
        self,
        original_tool_json: str,
        output_file: str,
        target_count: int,
        cos_th: float,
        bm25_th: float,
        model: str,
        temperature: float,
        max_tokens: int,
        progress=None
    ):
        """
        Tool variant generation main function
        
        Args:
            original_tool_json: Original tool JSON string
            output_file: Output file path
            target_count: Target generation count
            cos_th: Cosine similarity threshold
            bm25_th: BM25 similarity threshold
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum token count
            progress: Gradio progress object
        
        Returns:
            tuple: (status message, log message)
        """
        try:
            self.is_running = True
            self.current_output_file = output_file
            
            # ========== Step 1: Validate input ==========
            if progress:
                progress(0, desc="🔍 Validating input...")
            
            # Parse original tool
            try:
                original_tool = json.loads(original_tool_json)
                if not self._validate_tool_structure(original_tool):
                    return "❌ Error: Invalid tool structure, must contain name and description fields", ""
            except json.JSONDecodeError as e:
                return f"❌ Error: Invalid original tool JSON format - {str(e)}", ""
            
            # Create output directory
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            
            # ========== Step 2: Load existing tools ==========
            if progress:
                progress(0.1, desc="📚 Loading existing tools...")
            
            existing_tools = []
            try:
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                                normalized = self._normalize_tool(obj)
                                if normalized:
                                    existing_tools.append(normalized)
                            except Exception:
                                continue
                    print(f"📂 Loaded {len(existing_tools)} existing tools")
                else:
                    print("⚠️ Output file does not exist, will generate from scratch")
            except Exception as e:
                print(f"⚠️ Failed to load existing tools: {e}")
            
            # ========== Step 3: Initialize generator ==========
            if progress:
                progress(0.2, desc="🚀 Initializing generator...")
            
            generator = LLMGenerateLabel(
                self.api_keys, 
                self.api_base_url, 
                model, 
                temperature, 
                max_tokens
            )
            
            # ========== Step 4: Generate tool variants ==========
            if progress:
                progress(0.3, desc="🔄 Starting tool variant generation...")
            
            success_count = len(existing_tools)
            attempt_count = 0
            
            with open(output_file, "a", encoding="utf-8") as fout:
                while success_count < target_count:
                    attempt_count += 1
                    
                    # Generate new tool
                    new_tool = asyncio.run(self._generate_tool_variant(
                        generator, 
                        original_tool_json, 
                        existing_tools, 
                        attempt_count
                    ))
                    
                    if not new_tool:
                        self.generation_logs.append(f"❌ Generation {attempt_count} failed")
                        continue
                    
                    # Similarity check
                    if self.similarity_checker:
                        rejected, cos_score, bm_score, reason = self.similarity_checker.check_variant_similarity(
                            new_tool, existing_tools, cos_th, bm25_th
                        )
                        log_msg = f"Attempt {attempt_count}: Cosine={cos_score:.3f}, BM25={bm_score:.3f}"
                        
                        if rejected:
                            self.generation_logs.append(f"❌ {log_msg} - {reason}")
                            continue
                        else:
                            self.generation_logs.append(f"✅ {log_msg} - {reason}")
                    else:
                        self.generation_logs.append(f"✅ Generation {attempt_count} successful (skipped similarity check)")
                    
                    # Save new tool
                    existing_tools.append(new_tool)
                    fout.write(json.dumps(new_tool, ensure_ascii=False) + "\n")
                    fout.flush()
                    
                    success_count += 1
                    print(f"✅ Generation {attempt_count} successful ({success_count}/{target_count})")
                    
                    # Update progress
                    if progress:
                        progress_value = 0.3 + 0.6 * success_count / target_count
                        progress(progress_value, desc=f"🔄 Generated {success_count}/{target_count} tools")
                    
                    # Brief delay to avoid API limits
                    asyncio.run(asyncio.sleep(2))
            
            # ========== Step 5: Generate report ==========
            if progress:
                progress(1.0, desc="✅ Generation completed!")
            
            result_msg = f"""
✅ Tool variant generation completed!

📊 Statistics:
   - Target count: {target_count} items
   - Actually generated: {success_count} items
   - Total attempts: {attempt_count} times
   - Success rate: {success_count/attempt_count*100:.2f}%

⚙️ Generation parameters:
   - Model: {model}
   - Temperature: {temperature}
   - Max tokens: {max_tokens}
   - Cosine similarity threshold: {cos_th}
   - BM25 similarity threshold: {bm25_th}

📁 Output file: {output_file}

⏰ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            log_msg = self._generate_generation_log()
            
            return result_msg, log_msg
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error occurred during generation:\n{str(e)}\n\nDetailed information:\n{traceback.format_exc()}"
            return error_msg, ""
        
        finally:
            self.is_running = False
    
    def _validate_tool_structure(self, tool):
        """Validate tool structure"""
        if not isinstance(tool, dict):
            return False
        if "name" not in tool or "description" not in tool:
            return False
        return True
    
    def _normalize_tool(self, obj):
        """Normalize various structures to standard tool dict"""
        def is_valid_tool(d):
            return isinstance(d, dict) and "name" in d and "description" in d

        if isinstance(obj, dict):
            if "tool" in obj and isinstance(obj["tool"], dict):
                if is_valid_tool(obj["tool"]):
                    return obj["tool"]
            if is_valid_tool(obj):
                return obj
            if "title" in obj and "description" in obj:
                d = obj.copy()
                d["name"] = d.get("name") or d["title"]
                return d
            return None

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    norm = self._normalize_tool(item)
                    if norm:
                        return norm
            return None
        return None
    
    def _parse_llm_json_output(self, llm_output):
        """Parse LLM output and normalize"""
        try:
            clean = llm_output.strip()
            if clean.startswith("```json"):
                clean = clean[7:].strip()
            elif clean.startswith("```"):
                clean = clean[3:].strip()
            if clean.endswith("```"):
                clean = clean[:-3].strip()
            obj = json.loads(clean)
            return self._normalize_tool(obj)
        except Exception as e:
            print(f"⚠️ JSON parsing failed: {e}\nOriginal: {llm_output[:200]}...")
            return None
    
    def _format_existing_variants(self, tools):
        """Format existing variants"""
        if not tools:
            return "No variants available"
        lines = []
        for i, t in enumerate(tools, 1):
            if isinstance(t, dict):
                name = t.get("name", "")
                desc = t.get("description", "")
                lines.append(f"{i}. {name} - {desc[:80]}")
            else:
                lines.append(f"{i}. [Invalid item type: {type(t).__name__}]")
        return "\n".join(lines)
    
    async def _generate_tool_variant(self, generator, original_tool, existing_tools, call_id):
        """Generate tool variant"""
        variants_text = self._format_existing_variants(existing_tools)
        user_prompt = TOOL_GENERATE_USER.format(tool=original_tool, variants=variants_text)
        messages = [
            {"role": "system", "content": TOOL_GENERATE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        
        print(f"🚀 Calling generation {call_id} ...")
        res = await generator.call_llm_api(messages)
        
        if not res or not res.get("content"):
            print("⚠️ Returned empty result")
            return None
        
        tool = self._parse_llm_json_output(res["content"])
        if tool is None:
            print("⚠️ Empty after parsing/normalization")
            return None
        
        return tool
    
    def _generate_generation_log(self):
        """Generate generation log"""
        if not self.generation_logs:
            return "No generation logs available"
        
        log_text = f"📋 Generation logs (latest {len(self.generation_logs)} items)\n\n"
        log_text += "="*80 + "\n\n"
        
        for log in self.generation_logs:
            log_text += f"• {log}\n"
        
        return log_text
    
    # ========== File operation methods (standard implementation) ==========
    
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


# ===================== Similarity Detection Class =====================
class AdvancedSimilarityChecker:
    def __init__(self, model_name):
        if not SIMILARITY_AVAILABLE:
            raise ImportError("Similarity detection modules not installed")
        
        print(f"Loading vector model: {model_name}")
        self.model = SentenceTransformer(model_name)
        import torch
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("✅ Vector model loading completed")

    def extract_text(self, tool_json):
        if isinstance(tool_json, dict):
            name = tool_json.get("name", "")
            desc = tool_json.get("description", "")
            return f"{name}: {desc}"
        return str(tool_json)[:200]

    def cosine_similarity(self, v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(dot / norm) if norm else 0.0

    def check_variant_similarity(self, new_tool, existing_tools, cos_th=0.7, bm25_th=0.6):
        if not existing_tools:
            return False, 0, 0, "No historical tools, skip check"

        new_text = self.extract_text(new_tool)
        texts = [self.extract_text(t) for t in existing_tools]
        all_texts = [new_text] + texts
        embs = self.model.encode(all_texts, convert_to_numpy=True)
        new_emb, old_embs = embs[0], embs[1:]
        cos_scores = [self.cosine_similarity(new_emb, e) for e in old_embs]

        retriever = bm25s.BM25(corpus=texts)
        retriever.index(bm25s.tokenize(texts))
        query_tok = bm25s.tokenize([new_text])
        _, scores = retriever.retrieve(query_tok, k=len(texts))
        bm_scores = [s for s in scores[0]]

        avg_cos = np.mean(cos_scores)
        avg_bm = np.mean(bm_scores)
        cos_ok = avg_cos > cos_th
        bm_ok = avg_bm < bm25_th
        if cos_ok and bm_ok:
            return False, avg_cos, avg_bm, "Passed check"
        reason = []
        if not cos_ok:
            reason.append(f"Cosine similarity too low {avg_cos:.3f}")
        if not bm_ok:
            reason.append(f"BM25 similarity too high {avg_bm:.3f}")
        return True, avg_cos, avg_bm, "; ".join(reason)


# ===================== LLM Caller Class =====================
class LLMGenerateLabel:
    """Async GPT series caller"""

    def __init__(self, api_keys: List[str], api_base_url: str, model: str, temperature, max_tokens: int):
        self.api_base_url = api_base_url
        self.api_keys = api_keys
        self.api_key_cycle = itertools.cycle(api_keys)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._lock = asyncio.Lock()

    async def call_llm_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Async GPT API call with automatic key rotation and retry"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            async with self._lock:
                api_key = next(self.api_key_cycle)
            client = AsyncOpenAI(api_key=api_key, base_url=self.api_base_url)
            try:
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content
                return {"content": text}
            except Exception as e:
                print(f"⚠️ Call {attempt+1}/{max_retries} failed ({api_key[-4:]}): {e}")
                await asyncio.sleep(retry_delay)

        print("❌ All retries failed")
        return {"content": ""}
