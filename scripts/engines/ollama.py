"""
Ollama adapter for the LLM Testing Framework.

This module provides an adapter for the Ollama inference engine, using
the native /api/generate endpoint for detailed metrics.
"""

import json
import time
from typing import Dict, List, Optional, Union

import requests


class OllamaAdapter:
    """
    Adapter for Ollama inference engine.

    Uses the native /api/generate endpoint for streaming responses
    and detailed metrics extraction.
    """

    DEFAULT_URL = 'http://localhost:11434'

    def __init__(self, url: Optional[str] = None):
        """
        Initialize OllamaAdapter.

        Args:
            url: Ollama server URL (default: http://localhost:11434)
        """
        self.base_url = url or self.DEFAULT_URL
        self._current_model = None  # type: Optional[str]
        self._version = None  # type: Optional[str]

    def get_name(self) -> str:
        """Get the engine name."""
        return 'ollama'

    def get_version(self) -> str:
        """Get the Ollama version."""
        if self._version is not None:
            return self._version

        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self._version = data.get('version', 'unknown')
                return self._version
        except Exception:
            pass

        return 'unknown'

    def is_available(self) -> bool:
        """Check if Ollama is available and reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict]:
        """
        List available models.

        Returns:
            List of model dictionaries
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception:
            pass
        return []

    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is already downloaded.

        Args:
            model_name: Model name to check

        Returns:
            True if model is available locally
        """
        models = self.list_models()
        # Check both exact name and name without tag
        for m in models:
            name = m.get('name', '')
            if name == model_name or name.split(':')[0] == model_name.split(':')[0]:
                return True
        return False

    def get_running_models(self) -> List[Dict]:
        """
        Get list of models currently loaded in memory.

        Returns:
            List of running model dictionaries
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/ps",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
        except Exception:
            pass
        return []

    def load_model(
        self,
        model_name: str,
        download_timeout_sec: int = 1800,
        load_timeout_sec: int = 600
    ) -> Dict:
        """
        Load a model into memory.

        Downloads if needed, then loads into VRAM.

        Args:
            model_name: Model name (e.g., 'qwen3:4b')
            download_timeout_sec: Download timeout in seconds
            load_timeout_sec: VRAM load timeout in seconds

        Returns:
            Dictionary with load status and details
        """
        result = {
            'success': False,
            'model': model_name,
            'downloaded': False,
            'load_time_sec': 0,
            'error': None
        }

        # Check if model needs download
        needs_download = not self.is_model_available(model_name)

        if needs_download:
            print(f"  Downloading model: {model_name}")
            download_result = self._pull_model(model_name, download_timeout_sec)
            if not download_result['success']:
                result['error'] = download_result.get('error', 'Download failed')
                return result
            result['downloaded'] = True
        else:
            print(f"  Model already present, skipping download")

        # Load model into VRAM by running a simple query
        print(f"  Loading into VRAM (timeout: {load_timeout_sec}s)...")
        load_start = time.time()

        try:
            # Use generate endpoint with keep_alive to ensure model stays loaded
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': model_name,
                    'prompt': 'test',
                    'stream': False,
                    'options': {'num_predict': 1},
                    'keep_alive': '10m'
                },
                timeout=load_timeout_sec
            )

            if response.status_code == 200:
                result['success'] = True
                result['load_time_sec'] = time.time() - load_start
                self._current_model = model_name
                print(f"  Load complete ({result['load_time_sec']:.1f}s)")
            else:
                result['error'] = f"Load failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    result['error'] = error_data.get('error', result['error'])
                except Exception:
                    pass

        except requests.Timeout:
            result['error'] = f"Load timeout after {load_timeout_sec}s"
        except Exception as e:
            result['error'] = str(e)

        return result

    def _pull_model(
        self,
        model_name: str,
        timeout_sec: int = 1800
    ) -> Dict:
        """
        Pull (download) a model with progress reporting.

        Args:
            model_name: Model name to pull
            timeout_sec: Timeout in seconds

        Returns:
            Dictionary with pull status
        """
        result = {'success': False, 'error': None}

        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={'name': model_name, 'stream': True},
                stream=True,
                timeout=timeout_sec
            )

            if response.status_code != 200:
                result['error'] = f"Pull failed with status {response.status_code}"
                return result

            last_status = ''
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    status = data.get('status', '')

                    # Show progress
                    if 'total' in data and 'completed' in data:
                        total = data['total']
                        completed = data['completed']
                        pct = (completed / total * 100) if total > 0 else 0
                        print(f"\r  {status}: {pct:.1f}%", end='', flush=True)
                    elif status != last_status:
                        print(f"\r  {status}", end='', flush=True)
                        last_status = status

                    if data.get('error'):
                        result['error'] = data['error']
                        return result

                except json.JSONDecodeError:
                    continue

            print()  # New line after progress
            result['success'] = True

        except requests.Timeout:
            result['error'] = f"Download timeout after {timeout_sec}s"
        except Exception as e:
            result['error'] = str(e)

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

        Uses /api/generate endpoint for detailed metrics.

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

        # Retry logic with exponential backoff
        max_retries = 3
        backoff_sec = [1, 2, 4]

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        'model': self._current_model,
                        'prompt': prompt,
                        'system': system_prompt,
                        'stream': False,
                        'options': {
                            'num_predict': max_tokens,
                            'temperature': temperature
                        },
                        'keep_alive': '10m'
                    },
                    timeout=300  # 5 min timeout for generation
                )

                if response.status_code == 200:
                    data = response.json()

                    result['response'] = data.get('response', '')

                    # Extract metrics (times are in nanoseconds)
                    prompt_eval_count = data.get('prompt_eval_count', 0)
                    eval_count = data.get('eval_count', 0)
                    eval_duration = data.get('eval_duration', 0)
                    prompt_eval_duration = data.get('prompt_eval_duration', 0)

                    result['tokens_prompt'] = prompt_eval_count
                    result['tokens_generated'] = eval_count

                    # Calculate tokens per second
                    if eval_duration > 0:
                        # eval_duration is in nanoseconds
                        result['tokens_per_sec'] = eval_count / (eval_duration / 1e9)

                    # Time to first token (prompt processing time)
                    if prompt_eval_duration > 0:
                        result['time_to_first_token_ms'] = prompt_eval_duration / 1e6

                    return result

                else:
                    error_msg = f"Query failed with status {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', error_msg)
                    except Exception:
                        pass
                    result['error'] = error_msg

            except requests.Timeout:
                result['error'] = "Query timeout"
            except requests.ConnectionError:
                result['error'] = "Connection error"
            except Exception as e:
                result['error'] = str(e)

            # Retry with backoff
            if attempt < max_retries - 1:
                wait_time = backoff_sec[attempt]
                print(f"\n    [RETRY] Attempt {attempt + 1} failed, waiting {wait_time}s...")
                time.sleep(wait_time)

        return result

    def unload_model(self) -> bool:
        """
        Unload the current model from VRAM.

        Uses keep_alive: 0 to trigger immediate unload.

        Returns:
            True if successful
        """
        if self._current_model is None:
            return True

        try:
            # Send request with keep_alive: 0 to unload
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self._current_model,
                    'prompt': '',
                    'keep_alive': 0
                },
                timeout=30
            )

            # Verify unload by checking /api/ps
            time.sleep(2)  # Give it a moment
            running = self.get_running_models()

            if not running:
                self._current_model = None
                return True

            # Check if our model is still running
            for m in running:
                if m.get('name', '').startswith(self._current_model.split(':')[0]):
                    return False

            self._current_model = None
            return True

        except Exception as e:
            print(f"[WARN] Unload failed: {e}")
            return False

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a model.

        Args:
            model_name: Model name

        Returns:
            Model info dictionary or None
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={'name': model_name},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None
