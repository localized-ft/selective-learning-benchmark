# Original frozen preparation inputs

This directory preserves `config.json` and `requests.jsonl.gz` from the initial
preparation. The LiteLLM proxy rejected authentication and produced no judgments.
These files are inputs to `qwen32_luna.py prepare --direct`, not completed results.
The direct OpenRouter run reused the exact request bytes; completed results and
their updated producer hashes are under `../qwen32_openrouter/`.
