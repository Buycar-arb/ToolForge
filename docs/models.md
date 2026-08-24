# Working with models

Everything here was verified against live APIs, not guessed from documentation.

## Choosing a provider

The provider is inferred from the model id:

| model id looks like | goes to |
|---------------------|---------|
| `claude-*`, or contains `opus` / `sonnet` / `haiku` | the native Anthropic Messages API |
| anything else | an OpenAI-compatible `/chat/completions` endpoint |

Force it with a prefix when the guess is wrong:

```bash
GENERATION_MODEL=openai:claude-sonnet-5      # Claude behind an OpenAI-style gateway
JUDGE_MODEL=anthropic:qwen3-max              # Qwen behind an Anthropic-style endpoint
```

That second case is real — Aliyun's MaaS serves Qwen over an Anthropic-compatible
endpoint, and the prefix is what makes it work:

```bash
ANTHROPIC_API_KEY=sk-ws-…
ANTHROPIC_BASE_URL=https://<your-app>.cn-beijing.maas.aliyuncs.com/apps/anthropic
GENERATION_MODEL=anthropic:qwen3-max
```

Verify any combination before spending a run on it:

```bash
toolforge doctor
```

## Three API quirks the client handles for you

These are the reasons a naive OpenAI client fails against current models. All
three are handled in `toolforge/llm.py`; you should not have to think about them.

### 1. GPT-5 rejects `max_tokens`

```
400 Unsupported parameter: 'max_tokens' is not supported with this model.
Use 'max_completion_tokens' instead.
```

GPT-5 and the o-series want `max_completion_tokens`; plenty of self-hosted
servers only understand `max_tokens`. Neither name is universally safe.

`ParameterStyle` guesses from the model id and, if the endpoint disagrees,
switches and retries — once per client, then remembered. Both names work with a
model that accepts either, so a wrong guess costs one round trip and nothing else.

### 2. The Anthropic SDK dropped `temperature`

`anthropic>=1.0` removed `temperature` from `messages.create()`. Passing it
raises `TypeError`, which is not an HTTP error and would otherwise surface as a
hard failure. The client inspects the SDK signature at construction time and
omits the parameter when it is not accepted, so both SDK generations work.

### 3. Models ignore the fence you asked for

Stage 3's prompt asks for the dialogue in a ```json block. GPT-5.1 returns the
bare object instead. The original parser required the fence and therefore
rejected **every** GPT-5.1 response.

`parse_dialogue_json` now accepts a ```json block, a plain ``` block, or raw
JSON, with or without prose around it, and finds the outermost balanced object
while ignoring braces inside strings. `tests/test_parsing.py` covers all of it.

## Model choice, in practice

| role | what matters | reasonable picks |
|------|--------------|------------------|
| `GENERATION_MODEL` | instruction-following and long structured output — it has to emit a whole conversation as valid JSON | `gpt-5.1`, `claude-sonnet-5` |
| `JUDGE_MODEL` | judgement. It decides what enters your training set, so this is the wrong place to economise | `claude-opus-5`, `gpt-5.1` |
| stage 1 variants | creativity, at `temperature=1.0` | anything competent |

Mixing providers is normal and often the right call — generate with one, judge
with another:

```bash
OPENAI_API_KEY=sk-…
ANTHROPIC_API_KEY=sk-ant-…
GENERATION_MODEL=gpt-5.1
JUDGE_MODEL=claude-opus-5
```

## Keys and rotation

Several keys, comma-separated, are rotated round-robin:

```bash
OPENAI_API_KEY=sk-key-1,sk-key-2,sk-key-3
```

A key that fails `RETRY_ATTEMPTS` times (exponential backoff, capped at
`RETRY_MAX_DELAY`) is set aside and the next one takes over. Only when every key
is exhausted does the call return empty, and the pipeline records that as a
failed attempt rather than crashing the run.

## Local models

Anything vLLM serves works — it is just an OpenAI-compatible endpoint:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Qwen3-8B --port 8000 --served-model-name qwen3-8b
```

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY          # vLLM ignores it, but the client needs something
GENERATION_MODEL=qwen3-8b     # must match --served-model-name
```
