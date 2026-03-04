import os

def _load_api_keys():
    env_keys = os.getenv("API_KEYS", "")
    if env_keys:
        return [k.strip() for k in env_keys.split(",") if k.strip()]
    return []

API_KEYS = _load_api_keys()