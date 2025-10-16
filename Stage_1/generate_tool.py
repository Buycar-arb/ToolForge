
import json
import asyncio
import itertools
import numpy as np
from typing import List, Dict, Any
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# === Basic Configuration ===
# API Keys Configuration
# Option 1: Load from environment variable
import os
api_keys_env = os.getenv("API_KEYS", "")
if api_keys_env:
    API_KEYS = [key.strip() for key in api_keys_env.split(",") if key.strip()]
else:
    # Option 2: Load from file or use default empty list
    # You should set your API keys via environment variable or configuration file
    API_KEYS = [
        "your-api-key-1",
        "your-api-key-2",
        "your-api-key-3",
        # Add more API keys as needed
    ]

# API Base URL Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL = "gpt-4"
TEMPERATURE = 1
MAX_TOKENS = 8192
from tool_prompts import TOOL_GENERATE_USER, TOOL_GENERATE_SYSTEM


# Load text vector model and BM25
from sentence_transformers import SentenceTransformer
import bm25s

# Model path configuration - update this path to your local model
MODEL_PATH = os.getenv("SENTENCE_TRANSFORMER_MODEL_PATH", "./models/bge-m3")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Vector model loaded successfully")

# ===================== Async GPT Caller Class =====================
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

# ===================== Tool Parsing and Normalization =====================
def normalize_tool(obj):
    """Normalize various structures (list, dict, wrapped) to standard tool dict"""
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
                norm = normalize_tool(item)
                if norm:
                    return norm
        return None
    return None

def parse_llm_json_output(llm_output):
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
        return normalize_tool(obj)
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}\nOriginal: {llm_output[:200]}...")
        return None

# ===================== Similarity Detection =====================
class AdvancedSimilarityChecker:
    def __init__(self, model_name):
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

# ===================== Generation Logic =====================
def format_existing_variants(tools):
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

async def generate_tool_variant(generator_label, original_tool, existing_tools, call_id):
    variants_text = format_existing_variants(existing_tools)
    user_prompt = TOOL_GENERATE_USER.format(tool=original_tool, variants=variants_text)
    messages = [
        {"role": "system", "content": TOOL_GENERATE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    print(f"🚀 Calling generation {call_id} ...")
    res = await generator_label.call_llm_api(messages)
    if not res or not res.get("content"):
        print("⚠️ Returned empty result")
        return None
    tool = parse_llm_json_output(res["content"])
    if tool is None:
        print("⚠️ Empty after parsing/normalization")
        return None
    return tool

# ===================== Main Function =====================
async def main():
    # Output file configuration - update this path as needed
    output_file = os.getenv("OUTPUT_FILE", "./output/tool_variants.jsonl")
    # Example original tool - replace with your actual tool definition
    original_tool = {
        "name": "example_search",
        "description": "Example search tool for finding information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query content"
                },
                "category": {
                    "type": "string",
                    "description": "Search category"
                }
            },
            "required": ["query"]
        }
    }
    generator_label = LLMGenerateLabel(API_KEYS, API_BASE_URL, MODEL, TEMPERATURE, MAX_TOKENS)
    similarity_checker = AdvancedSimilarityChecker(MODEL_PATH)

    existing = []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    norm = normalize_tool(obj)
                    if norm:
                        existing.append(norm)
                except Exception:
                    continue
        print(f"📂 Loaded {len(existing)} variants")
    except FileNotFoundError:
        print("⚠️ File not found, starting from scratch")

    target = 20
    attempt = 0
    with open(output_file, "a", encoding="utf-8") as fout:
        while len(existing) < target:
            attempt += 1
            new_tool = await generate_tool_variant(generator_label, json.dumps(original_tool, ensure_ascii=False), existing, attempt)
            if not new_tool:
                continue
            rejected, cos, bm, reason = similarity_checker.check_variant_similarity(new_tool, existing)
            print(f"Similarity check: Cosine={cos:.3f}, BM25={bm:.3f}, Result={reason}")
            if not rejected:
                existing.append(new_tool)
                fout.write(json.dumps(new_tool, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"✅ Generation {attempt} successful ({len(existing)}/{target})")
            else:
                print(f"❌ Rejected: {reason}")
            await asyncio.sleep(2)
    print("🎉 All generation completed!")

if __name__ == "__main__":
    asyncio.run(main())
