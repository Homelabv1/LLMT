"""
vLLM adapter for the LLM Testing Framework.

This module provides an adapter for the vLLM inference engine,
using the OpenAI-compatible /v1/chat/completions endpoint.
"""

import time
from typing import Dict, List, Optional

import requests


class VLLMAdapter:
    """
    Adapter for vLLM inference engine.

    Uses the OpenAI-compatible API endpoints for queries.
    vLLM loads models at server startup via the --model flag.
    """

    DEFAULT_URL = 'http://localhost:8000'

    def __init__(self, url: Optional[str] = None):
        """
        Initialize VLLMAdapter.

        Args:
            url: vLLM server URL (default: http://localhost:8000)
        """
        self.base_url = url or self.DEFAULT_URL
        self._current_model = None  # type: Optional[str]
        self._version = None  # type: Optional[str]

    def get_name(self) -> str:
        """Get the engine name."""
        return 'vllm'

    def get_version(self) -> str:
        """Get the vLLM version."""
        if self._version is not None:
            return self._version

        try:
            response = requests.get(
                f"{self.base_url}/version",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self._version = data.get('version', 'unknown')
                return self._version
        except Exception:
            pass

        # Try health endpoint as fallback
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                self._version = 'available'
                return self._version
        except Exception:
            pass

        return 'unknown'

    def is_available(self) -> bool:
        """Check if vLLM server is available."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict]:
        """
        List available models via OpenAI-compatible endpoint.

        Returns:
            List of model dictionaries
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('data', [])
        except Exception:
            pass
        return []

    def get_loaded_model(self) -> Optional[str]:
        """
        Get the currently loaded model name.

        Returns:
            Model name or None
        """
        models = self.list_models()
        if models:
            return models[0].get('id')
        return None

    def wait_for_ready(self, timeout_sec: int = 120) -> bool:
        """
        Wait for the server to become ready.

        Args:
            timeout_sec: Maximum wait time in seconds

        Returns:
            True if server became ready
        """
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            if self.is_available():
                # Also check if model is loaded
                if self.get_loaded_model():
                    return True

            print(f"\r  Waiting for vLLM server...", end='', flush=True)
            time.sleep(5)

        return False

    def load_model(
        self,
        model_name: str,
        download_timeout_sec: int = 1800,
        load_timeout_sec: int = 600
    ) -> Dict:
        """
        'Load' a model - for vLLM this means waiting for the server.

        vLLM loads models at startup, so this method just waits
        for the server to become ready with the specified model.

        Args:
            model_name: Model name (for tracking purposes)
            download_timeout_sec: Not directly used for vLLM
            load_timeout_sec: Timeout for waiting for server readiness

        Returns:
            Dictionary with load status
        """
        result = {
            'success': False,
            'model': model_name,
            'downloaded': False,
            'load_time_sec': 0,
            'error': None
        }

        self._current_model = model_name
        print(f"  Waiting for vLLM server to be ready (timeout: {load_timeout_sec}s)...")

        start_time = time.time()
        if self.wait_for_ready(load_timeout_sec):
            result['success'] = True
            result['load_time_sec'] = time.time() - start_time

            # Verify correct model is loaded
            loaded = self.get_loaded_model()
            if loaded:
                self._current_model = loaded
                print(f"\n  Server ready with model: {loaded} ({result['load_time_sec']:.1f}s)")
            else:
                print(f"\n  Server ready ({result['load_time_sec']:.1f}s)")
        else:
            result['error'] = f"Server not ready after {load_timeout_sec}s"
            print(f"\n  {result['error']}")

        return result

    def query(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> Dict:
        """
        Query the model using OpenAI-compatible chat completions API.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dictionary with response and metrics
        """
        result = {
            'response': '',
            'tokens_prompt': 0,
            'tokens_generated': 0,
            'tokens_per_sec': 0,
            'time_to_first_token_ms': None,
            'error': None
        }

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]

        try:
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    'model': self._current_model,
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                },
                timeout=300  # 5 min timeout
            )

            total_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                # Extract response
                choices = data.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    result['response'] = message.get('content', '').strip()

                # Extract token counts from usage field
                usage = data.get('usage', {})
                result['tokens_prompt'] = usage.get('prompt_tokens', 0)
                result['tokens_generated'] = usage.get('completion_tokens', 0)

                # Calculate tokens per second
                if result['tokens_generated'] > 0 and total_time > 0:
                    result['tokens_per_sec'] = result['tokens_generated'] / total_time

            else:
                result['error'] = f"Query failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        result['error'] = error_data['error'].get('message', result['error'])
                except Exception:
                    pass

        except requests.Timeout:
            result['error'] = "Query timeout"
        except requests.ConnectionError:
            result['error'] = "Connection error"
        except Exception as e:
            result['error'] = str(e)

        return result

    def unload_model(self) -> bool:
        """
        Unload the model - not directly supported by vLLM.

        vLLM requires a server restart to change models.
        This method just clears the internal state.

        Returns:
            True (always succeeds as we just clear local state)
        """
        self._current_model = None
        return True

    def get_metrics(self) -> Optional[Dict]:
        """
        Get vLLM metrics from the Prometheus endpoint.

        Returns:
            Metrics dictionary or None
        """
        try:
            response = requests.get(
                f"{self.base_url}/metrics",
                timeout=10
            )
            if response.status_code == 200:
                # Parse Prometheus format (basic parsing)
                metrics = {}
                for line in response.text.split('\n'):
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            metrics[parts[0]] = parts[1]
                return metrics
        except Exception:
            pass
        return None

    def check_model_compatibility(self, model_name: str) -> Dict:
        """
        Check if a model is compatible with the current GPU setup.

        This is a basic check - actual compatibility depends on
        GPU memory, quantization, etc.

        Args:
            model_name: HuggingFace model ID

        Returns:
            Compatibility info dictionary
        """
        # Common model sizes (approximate, in billions of parameters)
        size_hints = {
            '0.5b': 0.5, '1b': 1, '1.5b': 1.5, '3b': 3, '4b': 4,
            '7b': 7, '8b': 8, '13b': 13, '14b': 14, '32b': 32,
            '70b': 70, '72b': 72
        }

        model_lower = model_name.lower()
        estimated_size = None

        for hint, size in size_hints.items():
            if hint in model_lower:
                estimated_size = size
                break

        # Very rough VRAM estimates (FP16)
        # ~2GB per billion parameters
        if estimated_size:
            estimated_vram_gb = estimated_size * 2

            return {
                'model': model_name,
                'estimated_params_b': estimated_size,
                'estimated_vram_fp16_gb': estimated_vram_gb,
                'notes': 'AWQ/GPTQ quantization can reduce VRAM by ~4x'
            }

        return {
            'model': model_name,
            'estimated_params_b': None,
            'notes': 'Could not estimate model size from name'
        }
