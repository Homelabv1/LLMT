"""
Core modules for the LLM Testing Framework.

This package provides the foundational components for testing language models
on homelab inventory queries.

Modules:
    config: Centralized configuration and defaults
    questions: Question loading and validation
    results: Results schema and JSONL writer
    metrics: GPU monitoring and cooldown logic
    runner: Test execution engine
    validation: Input validation utilities
    logging_config: Logging setup and utilities
"""

from .config import (
    DEFAULT_QUESTION_COUNT,
    DEFAULT_TIMEOUTS,
    DEFAULT_GPU_CONFIG,
    DEFAULT_TEST_CONFIG,
    OLLAMA_DEFAULT_URL,
    LLAMACPP_DEFAULT_URL,
    VLLM_DEFAULT_URL,
)
from .questions import (
    Question,
    load_questions,
    validate_questions,
    get_questions_by_category,
)
from .results import (
    QuestionResult,
    RunMetadata,
    ResultsWriter,
    load_results,
    get_completed_question_ids,
)
from .metrics import (
    GPUMetrics,
    GPUMonitor,
    get_gpu_info,
    wait_for_cooldown,
)
from .runner import (
    TestRunner,
    TestConfig,
    run_single_test,
)
from .validation import (
    ValidationError,
    validate_model_name,
    validate_model_name_strict,
    validate_file_path,
    sanitize_for_path,
)
from .logging_config import (
    setup_logging,
    get_logger,
)
from .preflight import (
    ModelStatus,
    OverallStatus,
    ModelCheck,
    ConnectivityCheck,
    PreflightResult,
    PreflightChecker,
    OllamaPreflightChecker,
    LlamaCppPreflightChecker,
    VLLMPreflightChecker,
    get_preflight_checker,
    run_preflight,
    detect_gpu,
)

__all__ = [
    # config
    'DEFAULT_QUESTION_COUNT',
    'DEFAULT_TIMEOUTS',
    'DEFAULT_GPU_CONFIG',
    'DEFAULT_TEST_CONFIG',
    'OLLAMA_DEFAULT_URL',
    'LLAMACPP_DEFAULT_URL',
    'VLLM_DEFAULT_URL',
    # questions
    'Question',
    'load_questions',
    'validate_questions',
    'get_questions_by_category',
    # results
    'QuestionResult',
    'RunMetadata',
    'ResultsWriter',
    'load_results',
    'get_completed_question_ids',
    # metrics
    'GPUMetrics',
    'GPUMonitor',
    'get_gpu_info',
    'wait_for_cooldown',
    # runner
    'TestRunner',
    'TestConfig',
    'run_single_test',
    # validation
    'ValidationError',
    'validate_model_name',
    'validate_model_name_strict',
    'validate_file_path',
    'sanitize_for_path',
    # logging
    'setup_logging',
    'get_logger',
    # preflight
    'ModelStatus',
    'OverallStatus',
    'ModelCheck',
    'ConnectivityCheck',
    'PreflightResult',
    'PreflightChecker',
    'OllamaPreflightChecker',
    'LlamaCppPreflightChecker',
    'VLLMPreflightChecker',
    'get_preflight_checker',
    'run_preflight',
    'detect_gpu',
]

__version__ = '1.0.0'
