"""
Centralized configuration for the LLM Testing Framework.

This module contains default values and configuration constants
used throughout the framework.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# =============================================================================
# Question/Test Configuration
# =============================================================================

# Default question count (should match the default questions file)
DEFAULT_QUESTION_COUNT = 100

# Default paths
DEFAULT_QUESTIONS_PATH = "scripts/data/homelab_test_questions.json"
DEFAULT_INVENTORY_PATH = "scripts/data/homelab_inventory.json"

# =============================================================================
# Engine Configuration
# =============================================================================

# Default engine URLs (can be overridden by environment variables)
OLLAMA_DEFAULT_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
LLAMACPP_DEFAULT_URL = os.environ.get('LLAMACPP_URL', 'http://localhost:8080')
VLLM_DEFAULT_URL = os.environ.get('VLLM_URL', 'http://localhost:8000')

# =============================================================================
# Timeout Configuration (in seconds)
# =============================================================================

@dataclass
class TimeoutConfig:
    """Timeout configuration for various operations."""
    download_sec: int = 1800    # Model download timeout (30 min)
    load_sec: int = 600         # Model load timeout (10 min)
    query_sec: int = 300        # Query timeout (5 min)
    health_check_sec: int = 10  # Health check timeout
    unload_sec: int = 30        # Model unload timeout


DEFAULT_TIMEOUTS = TimeoutConfig()

# =============================================================================
# GPU Configuration
# =============================================================================

@dataclass
class GPUConfig:
    """GPU-related configuration."""
    target_temp_c: int = 50         # Target temperature for cooldown
    max_temp_c: int = 70            # Maximum temperature threshold
    critical_temp_c: int = 85       # Critical temperature (abort)
    min_cooldown_sec: int = 30      # Minimum cooldown time
    max_cooldown_sec: int = 300     # Maximum cooldown time (5 min)
    vram_baseline_tolerance: float = 0.2  # 20% tolerance for VRAM baseline
    check_interval_sec: float = 2.0  # GPU metrics check interval


DEFAULT_GPU_CONFIG = GPUConfig()

# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Retry configuration for transient failures."""
    max_attempts: int = 3
    backoff_base_sec: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_sec: float = 30.0


DEFAULT_RETRY_CONFIG = RetryConfig()

# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestDefaults:
    """Default test configuration values."""
    max_tokens: int = 512
    temperature: float = 0.1
    warmup_queries: int = 1
    system_prompt_template: str = (
        "You are a helpful assistant that answers questions about "
        "a homelab inventory. Use the provided inventory data to "
        "give accurate, concise answers. If information is not "
        "available in the inventory, say so clearly."
    )


DEFAULT_TEST_CONFIG = TestDefaults()

# =============================================================================
# Model Name Validation
# =============================================================================

# Valid characters for model names (alphanumeric, dash, underscore, colon, dot, slash)
MODEL_NAME_PATTERN = r'^[a-zA-Z0-9._:/-]+$'
MODEL_NAME_MAX_LENGTH = 256

# =============================================================================
# File Paths
# =============================================================================

# Results file extensions
RESULTS_EXTENSION = '.jsonl'
SCORES_EXTENSION = '.scores.json'
METRICS_EXTENSION = '.metrics.csv'

# =============================================================================
# Logging Configuration
# =============================================================================

@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = os.environ.get('LLMT_LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    log_to_file: bool = False
    log_file_path: Optional[str] = None


DEFAULT_LOG_CONFIG = LogConfig()


def get_log_level_int(level_str: str) -> int:
    """Convert log level string to integer."""
    import logging
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)
