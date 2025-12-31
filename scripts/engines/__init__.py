"""
Engine adapters for the LLM Testing Framework.

This package provides adapters for different inference engines:
- Ollama: Easy-to-use curated models
- llama.cpp: Maximum flexibility with GGUF format
- vLLM: Production throughput with OpenAI-compatible API

Each adapter implements a common interface for model loading, querying,
and unloading.
"""

from typing import Any, Optional, Protocol, Dict


class EngineAdapter(Protocol):
    """Protocol defining the interface for engine adapters."""

    def get_name(self) -> str:
        """Get the engine name."""
        ...

    def get_version(self) -> str:
        """Get the engine version."""
        ...

    def is_available(self) -> bool:
        """Check if the engine is available and reachable."""
        ...

    def load_model(
        self,
        model_name: str,
        download_timeout_sec: int = 1800,
        load_timeout_sec: int = 600
    ) -> Dict:
        """
        Load a model into memory.

        Args:
            model_name: Name of the model to load
            download_timeout_sec: Timeout for model download
            load_timeout_sec: Timeout for loading into VRAM

        Returns:
            Dictionary with load status and details
        """
        ...

    def query(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict:
        """
        Query the model with a prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with response and metrics
        """
        ...

    def unload_model(self) -> bool:
        """
        Unload the current model from memory.

        Returns:
            True if successful
        """
        ...


def get_adapter(
    engine: str,
    url: Optional[str] = None
) -> Any:
    """
    Factory function to get an engine adapter.

    Args:
        engine: Engine name (ollama, llamacpp, vllm)
        url: Optional override URL for the engine

    Returns:
        Engine adapter instance

    Raises:
        ValueError: If engine is not recognized
    """
    engine = engine.lower()

    if engine == 'ollama':
        from .ollama import OllamaAdapter
        return OllamaAdapter(url=url)
    elif engine == 'llamacpp':
        from .llamacpp import LlamaCppAdapter
        return LlamaCppAdapter(url=url)
    elif engine == 'vllm':
        from .vllm import VLLMAdapter
        return VLLMAdapter(url=url)
    else:
        raise ValueError(f"Unknown engine: {engine}. Supported: ollama, llamacpp, vllm")


# Default URLs for each engine
DEFAULT_URLS = {
    'ollama': 'http://localhost:11434',
    'llamacpp': 'http://localhost:8080',
    'vllm': 'http://localhost:8000'
}


__all__ = [
    'EngineAdapter',
    'get_adapter',
    'DEFAULT_URLS'
]
