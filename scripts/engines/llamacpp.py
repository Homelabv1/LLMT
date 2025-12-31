"""
llama.cpp adapter for the LLM Testing Framework.

This module provides an adapter for the llama.cpp inference engine,
using the /completion endpoint for queries.

Note: llama.cpp loads models at server startup, so model switching
requires restarting the container. See llamacpp_batch.sh for the
container management workflow.
"""

import os
import time
from typing import Dict, Optional

import requests

# Import from core config if available, otherwise use defaults
try:
    from core.config import LLAMACPP_DEFAULT_URL
except ImportError:
    LLAMACPP_DEFAULT_URL = os.environ.get('LLAMACPP_URL', 'http://localhost:8080')


class LlamaCppAdapter:
    """
    Adapter for llama.cpp inference engine.

    llama.cpp loads models at server startup via the --model flag.
    This adapter assumes the model is already loaded and focuses
    on querying via the /completion endpoint.
    """

    DEFAULT_URL = LLAMACPP_DEFAULT_URL

    def __init__(self, url: Optional[str] = None):
        """
        Initialize LlamaCppAdapter.

        Args:
            url: llama.cpp server URL (default: from LLAMACPP_URL env or http://localhost:8080)
        """
        self.base_url = url or os.environ.get('LLAMACPP_URL', self.DEFAULT_URL)
        self._current_model = None  # type: Optional[str]

    def get_name(self) -> str:
        """Get the engine name."""
        return 'llamacpp'

    def get_version(self) -> str:
        """Get the llama.cpp version."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                # llama.cpp returns build info in health endpoint
                return data.get('version', 'unknown')
        except Exception:
            pass
        return 'unknown'

    def is_available(self) -> bool:
        """Check if llama.cpp server is available and healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', '')
                # Server is ready when status is "ok" or "ready"
                return status in ['ok', 'ready', 'no slot available']
            return False
        except Exception:
            return False

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
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', '')
                    if status in ['ok', 'ready']:
                        return True
                    elif status == 'loading model':
                        print(f"\r  Loading model...", end='', flush=True)
            except Exception:
                pass

            time.sleep(5)

        return False

    def load_model(
        self,
        model_name: str,
        download_timeout_sec: int = 1800,
        load_timeout_sec: int = 600
    ) -> Dict:
        """
        'Load' a model - for llama.cpp this means waiting for the server.

        Since llama.cpp loads models at startup, this method just waits
        for the server to become ready with the already-specified model.

        Args:
            model_name: Model name (for tracking purposes)
            download_timeout_sec: Not used for llama.cpp
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
        print(f"  Waiting for llama.cpp server to be ready (timeout: {load_timeout_sec}s)...")

        start_time = time.time()
        if self.wait_for_ready(load_timeout_sec):
            result['success'] = True
            result['load_time_sec'] = time.time() - start_time
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
        Query the model with a prompt.

        Uses the /completion endpoint.

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

        # Format prompt for llama.cpp
        # Using ChatML-style formatting for compatibility
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

        try:
            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/completion",
                json={
                    'prompt': full_prompt,
                    'n_predict': max_tokens,
                    'temperature': temperature,
                    'stop': ['<|im_end|>', '<|im_start|>'],
                    'stream': False
                },
                timeout=300  # 5 min timeout
            )

            total_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                result['response'] = data.get('content', '').strip()

                # Extract token counts
                result['tokens_prompt'] = data.get('tokens_evaluated', 0)
                result['tokens_generated'] = data.get('tokens_predicted', 0)

                # Calculate tokens per second
                timings = data.get('timings', {})
                if timings:
                    # predicted_per_second is tokens/sec for generation
                    result['tokens_per_sec'] = timings.get('predicted_per_second', 0)
                    # prompt_per_second could give us TTFT estimate
                    prompt_ms = timings.get('prompt_ms', 0)
                    if prompt_ms > 0:
                        result['time_to_first_token_ms'] = prompt_ms
                elif result['tokens_generated'] > 0:
                    result['tokens_per_sec'] = result['tokens_generated'] / total_time

            else:
                result['error'] = f"Query failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    result['error'] = error_data.get('error', result['error'])
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
        Unload the model - not directly supported by llama.cpp.

        llama.cpp requires a server restart to change models.
        This method just clears the internal state.

        Returns:
            True (always succeeds as we just clear local state)
        """
        self._current_model = None
        return True

    def get_props(self) -> Optional[Dict]:
        """
        Get server properties including model info.

        Returns:
            Server properties dictionary or None
        """
        try:
            response = requests.get(
                f"{self.base_url}/props",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def get_slots(self) -> Optional[Dict]:
        """
        Get slot information.

        Returns:
            Slots info dictionary or None
        """
        try:
            response = requests.get(
                f"{self.base_url}/slots",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
