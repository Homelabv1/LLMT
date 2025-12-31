"""
Core modules for the LLM Testing Framework.

This package provides the foundational components for testing language models
on homelab inventory queries.

Modules:
    questions: Question loading and validation
    results: Results schema and JSONL writer
    metrics: GPU monitoring and cooldown logic
    runner: Test execution engine
"""

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

__all__ = [
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
]

__version__ = '1.0.0'
