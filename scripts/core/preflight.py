"""
Pre-flight check system for the LLM Testing Framework.

This module provides verification of model availability and engine connectivity
before test execution, catching configuration errors early.

Levels of checks:
1. Engine Connectivity - Service running, API reachable
2. Model Availability - Valid names, cached locally, registry exists
3. Resource Feasibility - VRAM estimates, disk space, GPU access
4. Load Test (optional) - Actually load model and send test prompt
"""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

from .config import (
    OLLAMA_DEFAULT_URL,
    LLAMACPP_DEFAULT_URL,
    VLLM_DEFAULT_URL,
    DEFAULT_TIMEOUTS,
)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ModelStatus(Enum):
    """Status of a model after pre-flight check."""
    READY_CACHED = "ready_cached"
    READY_NEEDS_DOWNLOAD = "ready_needs_download"
    WARNING_TIGHT_VRAM = "warning_tight_vram"
    WARNING_NEEDS_DOWNLOAD = "warning_needs_download"
    FAILED_NOT_FOUND = "failed_not_found"
    FAILED_INVALID = "failed_invalid"
    FAILED_NO_SPACE = "failed_no_space"
    FAILED_AUTH = "failed_auth"
    FAILED_LOAD = "failed_load"
    FAILED_QUERY = "failed_query"
    UNKNOWN = "unknown"


class OverallStatus(Enum):
    """Overall status of pre-flight check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ModelCheck:
    """Result of checking a single model."""
    name: str
    status: ModelStatus
    cached: bool = False
    size_gb: Optional[float] = None
    vram_estimate_gb: Optional[float] = None
    fits_vram: bool = True
    vram_fit_status: str = "unknown"  # comfortable, tight_fit, too_large
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectivityCheck:
    """Result of connectivity check."""
    service_running: bool = False
    api_reachable: bool = False
    api_version: Optional[str] = None
    gpu_detected: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class DiskCheck:
    """Result of disk space check."""
    path: str = ""
    available_gb: float = 0.0
    required_gb: float = 0.0
    has_space: bool = True


@dataclass
class PreflightResult:
    """Complete result of pre-flight check."""
    engine: str
    engine_url: str
    timestamp: str
    connectivity: ConnectivityCheck
    models: List[ModelCheck]
    disk: DiskCheck
    can_proceed: bool
    overall_status: OverallStatus
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "engine": self.engine,
            "engine_url": self.engine_url,
            "timestamp": self.timestamp,
            "connectivity": {
                "service_running": self.connectivity.service_running,
                "api_reachable": self.connectivity.api_reachable,
                "api_version": self.connectivity.api_version,
                "gpu_detected": self.connectivity.gpu_detected,
                "gpu_name": self.connectivity.gpu_name,
                "gpu_vram_gb": self.connectivity.gpu_vram_gb,
                "error_message": self.connectivity.error_message,
            },
            "models": [
                {
                    "name": m.name,
                    "status": m.status.value,
                    "cached": m.cached,
                    "size_gb": m.size_gb,
                    "vram_estimate_gb": m.vram_estimate_gb,
                    "fits_vram": m.fits_vram,
                    "vram_fit_status": m.vram_fit_status,
                    "error_message": m.error_message,
                }
                for m in self.models
            ],
            "disk": {
                "path": self.disk.path,
                "available_gb": self.disk.available_gb,
                "required_gb": self.disk.required_gb,
                "has_space": self.disk.has_space,
            },
            "summary": {
                "total": len(self.models),
                "ready_cached": sum(1 for m in self.models if m.status == ModelStatus.READY_CACHED),
                "ready_needs_download": sum(1 for m in self.models if m.status in (
                    ModelStatus.READY_NEEDS_DOWNLOAD, ModelStatus.WARNING_NEEDS_DOWNLOAD
                )),
                "warnings": sum(1 for m in self.models if m.status.value.startswith("warning")),
                "failed": sum(1 for m in self.models if m.status.value.startswith("failed")),
            },
            "warnings": self.warnings,
            "errors": self.errors,
            "overall_status": self.overall_status.value,
            "can_proceed": self.can_proceed,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def detect_gpu() -> Dict[str, Any]:
    """Detect GPU using nvidia-smi."""
    result = {
        'detected': False,
        'name': None,
        'vram_gb': None,
        'vram_used_gb': None,
        'temperature_c': None,
    }

    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.used,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if proc.returncode == 0:
            line = proc.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                result['detected'] = True
                result['name'] = parts[0]
                result['vram_gb'] = float(parts[1]) / 1024
                result['vram_used_gb'] = float(parts[2]) / 1024
                result['temperature_c'] = int(parts[3])
    except Exception:
        pass

    return result


def check_disk_space(path: str, required_gb: float) -> DiskCheck:
    """Check if enough disk space is available."""
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)
        # Add 10% buffer
        has_space = available_gb >= required_gb * 1.1
        return DiskCheck(
            path=path,
            available_gb=round(available_gb, 2),
            required_gb=round(required_gb, 2),
            has_space=has_space
        )
    except Exception:
        return DiskCheck(path=path, available_gb=0, required_gb=required_gb, has_space=False)


def estimate_vram_gguf(file_size_gb: float) -> float:
    """Estimate VRAM for GGUF model (already quantized)."""
    # GGUF files are already quantized, VRAM ≈ file size + small overhead
    overhead = 0.3  # ~300MB for runtime buffers
    return file_size_gb + overhead


def estimate_vram_hf(params_billions: float, quantization: Optional[str] = None) -> float:
    """Estimate VRAM for HuggingFace model."""
    # Bytes per parameter by dtype
    if quantization in ('awq', 'gptq', 'int4'):
        bpp = 0.5  # 4-bit
    elif quantization in ('int8',):
        bpp = 1.0
    else:
        bpp = 2.0  # float16/bfloat16

    model_size_gb = params_billions * bpp
    overhead = 0.5  # CUDA overhead

    return model_size_gb + overhead


def check_vram_fit(estimate_gb: float, available_gb: float) -> Tuple[bool, str]:
    """Check if model fits in VRAM."""
    if estimate_gb > available_gb:
        return False, "too_large"
    elif estimate_gb > available_gb * 0.85:
        return True, "tight_fit"
    else:
        return True, "comfortable"


def parse_model_size(model_name: str) -> Optional[float]:
    """Parse model size in billions from name."""
    import re

    model_lower = model_name.lower()

    # Common size patterns
    patterns = [
        r'(\d+\.?\d*)b[^a-z]',  # 4b, 7b, 1.5b
        r'-(\d+\.?\d*)b',      # -4b, -7b
        r':(\d+\.?\d*)b',      # :4b, :7b
        r'(\d+)x(\d+)b',       # 8x7b (MoE)
    ]

    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            if 'x' in pattern:
                # MoE model - use total params estimate
                experts = int(match.group(1))
                per_expert = float(match.group(2))
                # Rough estimate: active params ≈ 2x single expert
                return per_expert * 2
            else:
                return float(match.group(1))

    return None


def detect_quantization(model_name: str) -> Optional[str]:
    """Detect quantization type from model name."""
    model_lower = model_name.lower()

    if 'awq' in model_lower:
        return 'awq'
    elif 'gptq' in model_lower:
        return 'gptq'
    elif 'int4' in model_lower:
        return 'int4'
    elif 'int8' in model_lower:
        return 'int8'
    elif any(q in model_lower for q in ['q4_k_m', 'q4_0', 'q4_1', 'q5_k_m', 'q6_k']):
        return 'gguf_quantized'
    elif '.gguf' in model_lower:
        return 'gguf'

    return None


# =============================================================================
# Base Preflight Checker
# =============================================================================

class PreflightChecker:
    """Base class for pre-flight checkers."""

    def __init__(
        self,
        engine: str,
        engine_url: str,
        gpu_vram_gb: Optional[float] = None,
        timeout: int = 30
    ):
        self.engine = engine
        self.engine_url = engine_url
        self.gpu_vram_gb = gpu_vram_gb
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def check_connectivity(self) -> ConnectivityCheck:
        """Check engine service connectivity."""
        raise NotImplementedError

    def check_model(self, model_name: str) -> ModelCheck:
        """Check single model availability."""
        raise NotImplementedError

    def load_test(self, model_name: str) -> ModelCheck:
        """Perform load test on model."""
        raise NotImplementedError

    def check_all(
        self,
        models: List[str],
        full: bool = False,
        skip_disk_check: bool = False,
        skip_vram_check: bool = False
    ) -> PreflightResult:
        """Run complete pre-flight check."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Check connectivity
        connectivity = self.check_connectivity()

        # Use detected GPU if not specified
        if self.gpu_vram_gb is None and connectivity.gpu_vram_gb:
            self.gpu_vram_gb = connectivity.gpu_vram_gb

        # Check each model
        model_checks = []
        total_download_size = 0.0

        for model_name in models:
            check = self.check_model(model_name)

            # Override VRAM check if requested
            if skip_vram_check:
                check.fits_vram = True
                check.vram_fit_status = "skipped"

            # Full load test if requested
            if full and check.status not in (
                ModelStatus.FAILED_NOT_FOUND,
                ModelStatus.FAILED_INVALID,
                ModelStatus.FAILED_AUTH
            ):
                load_result = self.load_test(model_name)
                if load_result.status in (ModelStatus.FAILED_LOAD, ModelStatus.FAILED_QUERY):
                    check = load_result

            model_checks.append(check)

            # Track download size
            if not check.cached and check.size_gb:
                total_download_size += check.size_gb

        # Disk check
        if skip_disk_check:
            disk = DiskCheck(path="/", available_gb=999, required_gb=total_download_size, has_space=True)
        else:
            disk = check_disk_space("/", total_download_size)

        # Determine overall status
        warnings = []
        errors = []

        if not connectivity.service_running:
            errors.append(f"{self.engine} service is not running")
        elif not connectivity.api_reachable:
            errors.append(f"{self.engine} API is not reachable")

        if not connectivity.gpu_detected:
            warnings.append("No GPU detected (will use CPU)")

        for check in model_checks:
            if check.status.value.startswith("failed"):
                errors.append(f"{check.name}: {check.error_message or check.status.value}")
            elif check.status.value.startswith("warning"):
                warnings.append(f"{check.name}: {check.vram_fit_status}")

        if not disk.has_space:
            errors.append(f"Insufficient disk space: need {disk.required_gb:.1f}GB, have {disk.available_gb:.1f}GB")

        # Determine if we can proceed
        can_proceed = len(errors) == 0 or all(
            "not found" in e or "invalid" in e for e in errors
        )

        if errors:
            overall_status = OverallStatus.FAILED
        elif warnings:
            overall_status = OverallStatus.WARNING
        else:
            overall_status = OverallStatus.PASSED

        return PreflightResult(
            engine=self.engine,
            engine_url=self.engine_url,
            timestamp=timestamp,
            connectivity=connectivity,
            models=model_checks,
            disk=disk,
            can_proceed=can_proceed,
            overall_status=overall_status,
            warnings=warnings,
            errors=errors
        )


# =============================================================================
# Ollama Preflight Checker
# =============================================================================

class OllamaPreflightChecker(PreflightChecker):
    """Preflight checker for Ollama engine."""

    def check_connectivity(self) -> ConnectivityCheck:
        """Check Ollama service connectivity."""
        result = ConnectivityCheck()

        # Check API version endpoint
        try:
            response = requests.get(
                f"{self.engine_url}/api/version",
                timeout=self.timeout
            )
            if response.status_code == 200:
                result.service_running = True
                result.api_reachable = True
                data = response.json()
                result.api_version = data.get('version', 'unknown')
            else:
                result.error_message = f"API returned status {response.status_code}"
        except requests.ConnectionError:
            result.error_message = "Connection refused"
        except requests.Timeout:
            result.error_message = "Connection timeout"
        except Exception as e:
            result.error_message = str(e)

        # Check GPU
        gpu = detect_gpu()
        result.gpu_detected = gpu['detected']
        result.gpu_name = gpu['name']
        result.gpu_vram_gb = gpu['vram_gb']

        return result

    def check_model(self, model_name: str) -> ModelCheck:
        """Check if Ollama model is available."""
        result = ModelCheck(name=model_name, status=ModelStatus.UNKNOWN)

        # Check if model is cached locally
        try:
            response = requests.get(
                f"{self.engine_url}/api/tags",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])

                for m in models:
                    name = m.get('name', '')
                    if name == model_name or name.split(':')[0] == model_name.split(':')[0]:
                        result.cached = True
                        # Get size from model info
                        size_bytes = m.get('size', 0)
                        result.size_gb = size_bytes / (1024**3) if size_bytes else None
                        break
        except Exception as e:
            self.logger.debug(f"Error checking local models: {e}")

        # Try to get model info (works for both local and registry)
        try:
            response = requests.post(
                f"{self.engine_url}/api/show",
                json={'name': model_name},
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                # Model is valid (either local or in registry)
                result.status = ModelStatus.READY_CACHED if result.cached else ModelStatus.READY_NEEDS_DOWNLOAD

                # Get size from details
                if 'details' in data:
                    details = data['details']
                    result.details = details

                # Estimate VRAM
                params = parse_model_size(model_name)
                quant = detect_quantization(model_name)

                if result.size_gb:
                    result.vram_estimate_gb = estimate_vram_gguf(result.size_gb)
                elif params:
                    result.vram_estimate_gb = estimate_vram_hf(params, quant)

                # Check VRAM fit
                if self.gpu_vram_gb and result.vram_estimate_gb:
                    fits, fit_status = check_vram_fit(result.vram_estimate_gb, self.gpu_vram_gb)
                    result.fits_vram = fits
                    result.vram_fit_status = fit_status

                    if not fits:
                        result.status = ModelStatus.FAILED_NO_SPACE
                        result.error_message = f"Model needs ~{result.vram_estimate_gb:.1f}GB, only {self.gpu_vram_gb:.1f}GB available"
                    elif fit_status == "tight_fit":
                        result.status = ModelStatus.WARNING_TIGHT_VRAM if result.cached else ModelStatus.WARNING_NEEDS_DOWNLOAD

            elif response.status_code == 404:
                result.status = ModelStatus.FAILED_NOT_FOUND
                result.error_message = "Model not found in Ollama library"
            else:
                result.status = ModelStatus.FAILED_INVALID
                result.error_message = f"API returned status {response.status_code}"

        except Exception as e:
            result.status = ModelStatus.FAILED_INVALID
            result.error_message = str(e)

        return result

    def load_test(self, model_name: str) -> ModelCheck:
        """Perform load test on Ollama model."""
        result = ModelCheck(name=model_name, status=ModelStatus.UNKNOWN)

        try:
            # Load model with simple prompt
            response = requests.post(
                f"{self.engine_url}/api/generate",
                json={
                    'model': model_name,
                    'prompt': 'Say OK',
                    'stream': False,
                    'options': {'num_predict': 10},
                },
                timeout=120  # 2 min for load + generate
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')

                if response_text:
                    result.status = ModelStatus.READY_CACHED
                    result.cached = True
                else:
                    result.status = ModelStatus.FAILED_QUERY
                    result.error_message = "Model loaded but returned empty response"
            else:
                result.status = ModelStatus.FAILED_LOAD
                try:
                    error_data = response.json()
                    result.error_message = error_data.get('error', f"Status {response.status_code}")
                except Exception:
                    result.error_message = f"Status {response.status_code}"

        except requests.Timeout:
            result.status = ModelStatus.FAILED_LOAD
            result.error_message = "Load timeout"
        except Exception as e:
            result.status = ModelStatus.FAILED_LOAD
            result.error_message = str(e)

        # Unload model after test
        try:
            requests.post(
                f"{self.engine_url}/api/generate",
                json={'model': model_name, 'prompt': '', 'keep_alive': 0},
                timeout=10
            )
        except Exception:
            pass

        return result


# =============================================================================
# llama.cpp Preflight Checker
# =============================================================================

class LlamaCppPreflightChecker(PreflightChecker):
    """Preflight checker for llama.cpp engine."""

    def __init__(self, models_dir: str = "/models", **kwargs):
        super().__init__(**kwargs)
        self.models_dir = models_dir

    def check_connectivity(self) -> ConnectivityCheck:
        """Check llama.cpp server connectivity."""
        result = ConnectivityCheck()

        # Check health endpoint
        try:
            response = requests.get(
                f"{self.engine_url}/health",
                timeout=self.timeout
            )
            if response.status_code == 200:
                result.service_running = True
                result.api_reachable = True
                data = response.json()
                result.api_version = data.get('version', 'available')
            else:
                result.service_running = True
                result.error_message = f"Health check returned status {response.status_code}"
        except requests.ConnectionError:
            result.error_message = "Connection refused - server may not be running"
        except requests.Timeout:
            result.error_message = "Connection timeout"
        except Exception as e:
            result.error_message = str(e)

        # Check GPU
        gpu = detect_gpu()
        result.gpu_detected = gpu['detected']
        result.gpu_name = gpu['name']
        result.gpu_vram_gb = gpu['vram_gb']

        return result

    def check_model(self, model_path: str) -> ModelCheck:
        """Check if GGUF model file exists."""
        result = ModelCheck(name=model_path, status=ModelStatus.UNKNOWN)

        # Resolve path
        if model_path.startswith('/'):
            full_path = model_path
        else:
            full_path = os.path.join(self.models_dir, model_path)

        # Check if file exists
        if os.path.exists(full_path):
            result.cached = True

            # Get file size
            try:
                size_bytes = os.path.getsize(full_path)
                result.size_gb = size_bytes / (1024**3)
                result.vram_estimate_gb = estimate_vram_gguf(result.size_gb)

                # Check VRAM fit
                if self.gpu_vram_gb and result.vram_estimate_gb:
                    fits, fit_status = check_vram_fit(result.vram_estimate_gb, self.gpu_vram_gb)
                    result.fits_vram = fits
                    result.vram_fit_status = fit_status

                    if not fits:
                        result.status = ModelStatus.FAILED_NO_SPACE
                        result.error_message = f"Model needs ~{result.vram_estimate_gb:.1f}GB, only {self.gpu_vram_gb:.1f}GB available"
                    elif fit_status == "tight_fit":
                        result.status = ModelStatus.WARNING_TIGHT_VRAM
                    else:
                        result.status = ModelStatus.READY_CACHED
                else:
                    result.status = ModelStatus.READY_CACHED

            except Exception as e:
                result.status = ModelStatus.FAILED_INVALID
                result.error_message = f"Error reading file: {e}"
        else:
            result.status = ModelStatus.FAILED_NOT_FOUND
            result.error_message = f"File not found: {full_path}"

            # Check if it's a HuggingFace reference
            if '/' in model_path and not model_path.startswith('/'):
                result.error_message += " (may need to download from HuggingFace)"

        return result

    def load_test(self, model_path: str) -> ModelCheck:
        """Load test is not directly supported - llama.cpp loads at startup."""
        # Just return the regular check result
        return self.check_model(model_path)


# =============================================================================
# vLLM Preflight Checker
# =============================================================================

class VLLMPreflightChecker(PreflightChecker):
    """Preflight checker for vLLM engine."""

    def check_connectivity(self) -> ConnectivityCheck:
        """Check vLLM server connectivity."""
        result = ConnectivityCheck()

        # Try health endpoint
        try:
            response = requests.get(
                f"{self.engine_url}/health",
                timeout=self.timeout
            )
            if response.status_code == 200:
                result.service_running = True
                result.api_reachable = True
        except Exception:
            pass

        # Try version endpoint
        if not result.api_reachable:
            try:
                response = requests.get(
                    f"{self.engine_url}/version",
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result.service_running = True
                    result.api_reachable = True
                    data = response.json()
                    result.api_version = data.get('version', 'unknown')
            except Exception:
                pass

        # Try v1/models endpoint (OpenAI compatible)
        if not result.api_reachable:
            try:
                response = requests.get(
                    f"{self.engine_url}/v1/models",
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result.service_running = True
                    result.api_reachable = True
                    result.api_version = "available"
            except requests.ConnectionError:
                result.error_message = "Connection refused - server may not be running"
            except requests.Timeout:
                result.error_message = "Connection timeout"
            except Exception as e:
                result.error_message = str(e)

        # Check GPU
        gpu = detect_gpu()
        result.gpu_detected = gpu['detected']
        result.gpu_name = gpu['name']
        result.gpu_vram_gb = gpu['vram_gb']

        return result

    def check_model(self, model_name: str) -> ModelCheck:
        """Check if HuggingFace model is available."""
        result = ModelCheck(name=model_name, status=ModelStatus.UNKNOWN)

        # Try to check if model exists via HuggingFace API
        # For now, we'll do a basic check on the model name format

        # Check model name format
        if '/' not in model_name and not model_name.startswith('local/'):
            result.status = ModelStatus.FAILED_INVALID
            result.error_message = "vLLM models should be HuggingFace IDs (e.g., 'Qwen/Qwen2.5-3B-Instruct')"
            return result

        # Parse model size and quantization
        params = parse_model_size(model_name)
        quant = detect_quantization(model_name)

        if params:
            result.vram_estimate_gb = estimate_vram_hf(params, quant)

        # Check VRAM fit
        if self.gpu_vram_gb and result.vram_estimate_gb:
            fits, fit_status = check_vram_fit(result.vram_estimate_gb, self.gpu_vram_gb)
            result.fits_vram = fits
            result.vram_fit_status = fit_status

            if not fits:
                result.status = ModelStatus.FAILED_NO_SPACE
                result.error_message = f"Model needs ~{result.vram_estimate_gb:.1f}GB, only {self.gpu_vram_gb:.1f}GB available"
                return result

        # Try to check HuggingFace API
        try:
            hf_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(hf_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                result.cached = False  # We can't easily check local HF cache

                # Check for gated model
                if data.get('gated'):
                    # Check for HF_TOKEN
                    if not os.environ.get('HF_TOKEN') and not os.environ.get('HUGGING_FACE_HUB_TOKEN'):
                        result.status = ModelStatus.FAILED_AUTH
                        result.error_message = "Gated model requires HF_TOKEN environment variable"
                        return result

                if result.vram_fit_status == "tight_fit":
                    result.status = ModelStatus.WARNING_NEEDS_DOWNLOAD
                else:
                    result.status = ModelStatus.READY_NEEDS_DOWNLOAD

            elif response.status_code == 404:
                result.status = ModelStatus.FAILED_NOT_FOUND
                result.error_message = "Model not found on HuggingFace"
            else:
                # Can't verify, assume it exists
                result.status = ModelStatus.READY_NEEDS_DOWNLOAD
                result.details['hf_check'] = 'skipped'

        except Exception as e:
            # Can't connect to HuggingFace, assume model exists
            result.status = ModelStatus.READY_NEEDS_DOWNLOAD
            result.details['hf_check'] = f'failed: {e}'

        return result

    def load_test(self, model_name: str) -> ModelCheck:
        """vLLM load test - check if model is loaded via v1/models."""
        result = self.check_model(model_name)

        # Check if model is already loaded
        try:
            response = requests.get(
                f"{self.engine_url}/v1/models",
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                for m in models:
                    if m.get('id') == model_name:
                        result.status = ModelStatus.READY_CACHED
                        result.cached = True
                        break
        except Exception:
            pass

        return result


# =============================================================================
# Factory Function
# =============================================================================

def get_preflight_checker(
    engine: str,
    url: Optional[str] = None,
    gpu_vram_gb: Optional[float] = None,
    timeout: int = 30,
    **kwargs
) -> PreflightChecker:
    """Factory function to get appropriate preflight checker."""

    # Default URLs
    default_urls = {
        'ollama': OLLAMA_DEFAULT_URL,
        'llamacpp': LLAMACPP_DEFAULT_URL,
        'vllm': VLLM_DEFAULT_URL,
    }

    engine_url = url or default_urls.get(engine)

    if engine == 'ollama':
        return OllamaPreflightChecker(
            engine=engine,
            engine_url=engine_url,
            gpu_vram_gb=gpu_vram_gb,
            timeout=timeout
        )
    elif engine == 'llamacpp':
        return LlamaCppPreflightChecker(
            engine=engine,
            engine_url=engine_url,
            gpu_vram_gb=gpu_vram_gb,
            timeout=timeout,
            **kwargs
        )
    elif engine == 'vllm':
        return VLLMPreflightChecker(
            engine=engine,
            engine_url=engine_url,
            gpu_vram_gb=gpu_vram_gb,
            timeout=timeout
        )
    else:
        raise ValueError(f"Unknown engine: {engine}")


def run_preflight(
    engine: str,
    models: List[str],
    url: Optional[str] = None,
    full: bool = False,
    skip_disk_check: bool = False,
    skip_vram_check: bool = False,
    timeout: int = 30,
    **kwargs
) -> PreflightResult:
    """Convenience function to run pre-flight check."""
    checker = get_preflight_checker(engine, url, timeout=timeout, **kwargs)
    return checker.check_all(
        models,
        full=full,
        skip_disk_check=skip_disk_check,
        skip_vram_check=skip_vram_check
    )
