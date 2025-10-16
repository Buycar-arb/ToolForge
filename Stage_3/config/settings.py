# API config
import os

DEFAULT_MODEL = "anthropic.claude-sonnet-4"
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_DELAY = 40
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
RAG_MIN_TOP_K = 5
RAG_MAX_TOP_K = 10

# Tool Library and General Search Tool Configuration Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TOOL_BANK_DIR = os.path.join(BASE_DIR, "Stage_3", "tool_bank", "tools")
TARGET_GENERAL_FILE = os.path.join(TOOL_BANK_DIR, "general_information_search.jsonl")
