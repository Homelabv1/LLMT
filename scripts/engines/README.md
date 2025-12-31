# Engine Adapters

This package provides adapters for different LLM inference engines.

## Supported Engines

| Engine | Default Port | Model Format | Features |
|--------|-------------|--------------|----------|
| Ollama | 11434 | Ollama registry | Easy setup, curated models |
| llama.cpp | 8080 | GGUF | Maximum flexibility |
| vLLM | 8000 | HuggingFace | Production throughput |

## Adapter Protocol

All adapters implement this interface:

```python
class EngineAdapter(Protocol):
    def get_name(self) -> str:
        """Get the engine name."""

    def get_version(self) -> str:
        """Get the engine version."""

    def is_available(self) -> bool:
        """Check if the engine is reachable."""

    def load_model(
        self,
        model_name: str,
        download_timeout_sec: int = 1800,
        load_timeout_sec: int = 600
    ) -> dict:
        """Load a model into memory."""

    def query(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> dict:
        """Query the model."""

    def unload_model(self) -> bool:
        """Unload the current model."""
```

## Usage

```python
from engines import get_adapter

# Get adapter by name
adapter = get_adapter('ollama', url='http://localhost:11434')

# Check availability
if adapter.is_available():
    # Load model
    result = adapter.load_model('qwen3:4b')

    if result['success']:
        # Query
        response = adapter.query(
            prompt="What is the IP of pve-main?",
            system_prompt="You are a homelab assistant...",
            max_tokens=512,
            temperature=0.1
        )
        print(response['response'])

        # Unload
        adapter.unload_model()
```

## Engine-Specific Notes

### Ollama

- Uses native `/api/generate` endpoint
- Supports streaming download progress
- Unload via `keep_alive: 0`
- Retry logic with exponential backoff

```python
adapter = OllamaAdapter(url='http://localhost:11434')
```

### llama.cpp

- Model loaded at server startup
- Requires container restart to switch models
- Use `llamacpp_batch.sh` for batch testing
- Health check via `/health`

```python
adapter = LlamaCppAdapter(url='http://localhost:8080')
```

### vLLM

- OpenAI-compatible API
- Model loaded at server startup
- Supports tensor parallelism
- Uses `/v1/chat/completions`

```python
adapter = VLLMAdapter(url='http://localhost:8000')
```

## Adding a New Engine

1. Create `engines/myengine.py`:

```python
class MyEngineAdapter:
    def __init__(self, url: Optional[str] = None):
        self.base_url = url or 'http://localhost:8000'

    def get_name(self) -> str:
        return 'myengine'

    def get_version(self) -> str:
        # Query version endpoint
        ...

    def is_available(self) -> bool:
        # Health check
        ...

    def load_model(self, model_name, download_timeout_sec, load_timeout_sec):
        # Load model logic
        ...

    def query(self, prompt, system_prompt, max_tokens, temperature):
        # Query logic
        ...

    def unload_model(self):
        # Unload logic
        ...
```

2. Update `engines/__init__.py`:

```python
def get_adapter(engine: str, url: Optional[str] = None):
    if engine == 'myengine':
        from .myengine import MyEngineAdapter
        return MyEngineAdapter(url=url)
    # ...
```

## Response Format

Query responses should include:

```python
{
    'response': str,           # Generated text
    'tokens_prompt': int,      # Prompt tokens
    'tokens_generated': int,   # Generated tokens
    'tokens_per_sec': float,   # Generation speed
    'time_to_first_token_ms': float,  # Optional TTFT
    'error': str or None       # Error message if failed
}
```
