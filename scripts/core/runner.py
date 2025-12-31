"""
Test execution engine for the LLM Testing Framework.

This module provides the TestRunner class and supporting utilities for
running tests against language models, handling resumption, and graceful
shutdown.
"""

import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from .questions import Question, create_system_prompt, load_inventory, load_questions
from .results import (
    QuestionResult,
    ResultsWriter,
    RunMetadata,
    append_failure_log,
    get_completed_question_ids,
    write_load_failure,
)
from .metrics import (
    GPUMetrics,
    GPUMonitor,
    PowerMonitor,
    get_gpu_info,
    get_gpu_metrics,
    verify_vram_cleared,
    wait_for_cooldown,
)


class TestConfig:
    """Configuration for a test run."""

    def __init__(
        self,
        engine: str,
        model: str,
        gpu: str = '1660super-1x',
        url: Optional[str] = None,
        questions_file: Optional[str] = None,
        inventory_file: Optional[str] = None,
        results_dir: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        warmup_queries: int = 1,
        download_timeout: int = 1800,
        load_timeout: int = 600,
        system_prompt: Optional[str] = None,
        min_cooldown: int = 30,
        target_temp: int = 50,
        power_limit: Optional[str] = None,
        power_threshold: Optional[int] = None
    ):
        """
        Initialize TestConfig.

        Args:
            engine: Inference engine name (ollama, llamacpp, vllm)
            model: Model name
            gpu: GPU configuration name
            url: Override server URL
            questions_file: Path to questions JSON
            inventory_file: Path to inventory JSON
            results_dir: Base results directory
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            warmup_queries: Number of warmup queries
            download_timeout: Download timeout in seconds
            load_timeout: Load timeout in seconds
            system_prompt: System prompt override
            min_cooldown: Minimum cooldown time in seconds
            target_temp: Target temperature for cooldown
            power_limit: Power limits (e.g., "280,160")
            power_threshold: Power threshold in watts
        """
        self.engine = engine
        self.model = model
        self.gpu = gpu
        self.url = url
        self.questions_file = questions_file
        self.inventory_file = inventory_file
        self.results_dir = results_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.warmup_queries = warmup_queries
        self.download_timeout = download_timeout
        self.load_timeout = load_timeout
        self.system_prompt = system_prompt
        self.min_cooldown = min_cooldown
        self.target_temp = target_temp
        self.power_limit = power_limit
        self.power_threshold = power_threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'engine': self.engine,
            'model': self.model,
            'gpu': self.gpu,
            'url': self.url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'warmup_queries': self.warmup_queries,
            'download_timeout': self.download_timeout,
            'load_timeout': self.load_timeout,
            'min_cooldown': self.min_cooldown,
            'target_temp': self.target_temp,
            'power_limit': self.power_limit,
            'power_threshold': self.power_threshold
        }


class TestRunner:
    """
    Main test execution engine.

    Handles running tests with resumption support, graceful shutdown,
    and GPU metrics collection.
    """

    def __init__(self, config: TestConfig, engine_adapter: Any):
        """
        Initialize TestRunner.

        Args:
            config: Test configuration
            engine_adapter: Engine adapter instance (implements query method)
        """
        self.config = config
        self.engine = engine_adapter

        # State
        self._shutdown_requested = False
        self._force_shutdown = False
        self._current_question = None  # type: Optional[Question]

        # Components
        self.writer = None  # type: Optional[ResultsWriter]
        self.gpu_monitor = None  # type: Optional[GPUMonitor]
        self.power_monitor = None  # type: Optional[PowerMonitor]

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        if self._shutdown_requested:
            print("\n[FORCE] Second interrupt - forcing exit...")
            self._force_shutdown = True
            if self.gpu_monitor:
                self.gpu_monitor.stop()
            if self.power_monitor:
                self.power_monitor.stop()
            sys.exit(1)
        else:
            print("\n[SHUTDOWN] Finishing current question, then exiting...")
            print("[SHUTDOWN] Press Ctrl+C again to force immediate exit")
            self._shutdown_requested = True

    def run(
        self,
        questions: List[Question],
        inventory: Dict,
        completed_ids: Optional[Set[int]] = None
    ) -> Dict:
        """
        Run tests on the given questions.

        Args:
            questions: List of questions to test
            inventory: Inventory data
            completed_ids: Set of already completed question IDs to skip

        Returns:
            Dictionary with test results summary
        """
        if completed_ids is None:
            completed_ids = set()

        # Filter to remaining questions
        remaining = [q for q in questions if q.q_id not in completed_ids]

        if not remaining:
            print("All questions already completed!")
            return {'status': 'complete', 'questions_run': 0}

        # Initialize results writer
        self.writer = ResultsWriter(self.config.results_dir, self.config.model)

        # Create system prompt
        system_prompt = self.config.system_prompt
        if system_prompt is None:
            system_prompt = create_system_prompt(inventory)

        # Get GPU info
        gpu_info = get_gpu_info()
        gpu_model = gpu_info['name'] if gpu_info else 'Unknown'
        gpu_vram = gpu_info['vram_total_mb'] if gpu_info else 0

        # Create run metadata
        test_id = self._generate_test_id()
        metadata = RunMetadata(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            model_name=self.config.model,
            model_quantization=self._detect_quantization(),
            engine_name=self.config.engine,
            engine_version=self.engine.get_version(),
            gpu_name=self.config.gpu,
            gpu_model=gpu_model,
            gpu_vram_mb=gpu_vram,
            gpu_count=self._get_gpu_count(),
            test_config=self.config.to_dict(),
            power_limits_applied=self.config.power_limit,
            total_questions=len(questions)
        )
        self.writer.write_metadata(metadata)

        # Start GPU monitoring
        self.gpu_monitor = GPUMonitor(interval_sec=1.0)
        self.writer.write_gpu_metrics_header()

        # Set up metrics callback
        def metrics_callback(metrics: List[GPUMetrics], phase: str) -> None:
            if metrics and self.writer:
                m = metrics[0]  # Primary GPU
                self.writer.append_gpu_metrics(
                    timestamp=m.timestamp,
                    vram_used_mb=m.vram_used_mb,
                    vram_total_mb=m.vram_total_mb,
                    gpu_util_pct=m.gpu_util_pct,
                    temp_c=m.temp_c,
                    phase=phase
                )

        self.gpu_monitor.callback = metrics_callback
        self.gpu_monitor.phase = 'idle'
        self.gpu_monitor.start()

        # Start power monitoring if threshold set
        if self.config.power_threshold:
            self.power_monitor = PowerMonitor(
                threshold_watts=self.config.power_threshold
            )
            self.power_monitor.start()

        # Run warmup queries
        if self.config.warmup_queries > 0:
            print(f"Running {self.config.warmup_queries} warmup queries...")
            self._run_warmup(system_prompt)

        # Run test questions
        results_summary = {
            'status': 'incomplete',
            'questions_total': len(questions),
            'questions_completed': len(completed_ids),
            'questions_run': 0,
            'questions_failed': 0,
            'avg_latency_ms': 0,
            'avg_tokens_per_sec': 0,
            'shutdown_requested': False,
            'power_aborted': False
        }

        latencies = []
        tokens_per_sec = []

        try:
            for i, question in enumerate(remaining):
                if self._shutdown_requested or self._force_shutdown:
                    results_summary['shutdown_requested'] = True
                    break

                if self.power_monitor and self.power_monitor.should_abort:
                    print(f"\n[POWER] Aborting model due to power threshold")
                    results_summary['power_aborted'] = True
                    break

                self._current_question = question
                q_num = len(completed_ids) + i + 1

                print(f"[{q_num}/{len(questions)}] q{question.q_id}: {question.question[:50]}...", end=' ')

                self.gpu_monitor.phase = 'inference'

                try:
                    result = self._run_question(question, system_prompt)
                    self.writer.write_result(result)

                    if result.error:
                        print(f"ERROR ({result.error})")
                        results_summary['questions_failed'] += 1
                    else:
                        print(f"OK ({result.latency_ms:.0f}ms, {result.tokens_per_sec:.1f} t/s)")
                        latencies.append(result.latency_ms)
                        tokens_per_sec.append(result.tokens_per_sec)

                    results_summary['questions_run'] += 1
                    results_summary['questions_completed'] += 1

                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    results_summary['questions_failed'] += 1
                    results_summary['questions_run'] += 1

                    # Save error result
                    error_result = QuestionResult(
                        q_id=question.q_id,
                        category=question.category,
                        question=question.question,
                        expected=question.expected,
                        response='',
                        latency_ms=0,
                        tokens_prompt=0,
                        tokens_generated=0,
                        tokens_per_sec=0,
                        error=str(e)
                    )
                    self.writer.write_result(error_result)

                self.gpu_monitor.phase = 'idle'
                self._current_question = None

        finally:
            # Stop monitors
            if self.gpu_monitor:
                self.gpu_monitor.stop()
                summary = self.gpu_monitor.get_summary()
                self.writer.write_gpu_summary(summary)

            if self.power_monitor:
                self.power_monitor.stop()

        # Calculate averages
        if latencies:
            results_summary['avg_latency_ms'] = sum(latencies) / len(latencies)
        if tokens_per_sec:
            results_summary['avg_tokens_per_sec'] = sum(tokens_per_sec) / len(tokens_per_sec)

        # Update status
        if results_summary['questions_completed'] >= len(questions):
            results_summary['status'] = 'complete'
        elif self._shutdown_requested:
            results_summary['status'] = 'interrupted'
        elif results_summary['power_aborted']:
            results_summary['status'] = 'power_aborted'

        return results_summary

    def _run_question(self, question: Question, system_prompt: str) -> QuestionResult:
        """
        Run a single question and return the result.

        Args:
            question: The question to run
            system_prompt: System prompt to use

        Returns:
            QuestionResult with response data
        """
        start_time = time.time()

        response = self.engine.query(
            prompt=question.question,
            system_prompt=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        latency_ms = (time.time() - start_time) * 1000

        return QuestionResult(
            q_id=question.q_id,
            category=question.category,
            question=question.question,
            expected=question.expected,
            response=response.get('response', ''),
            latency_ms=latency_ms,
            tokens_prompt=response.get('tokens_prompt', 0),
            tokens_generated=response.get('tokens_generated', 0),
            tokens_per_sec=response.get('tokens_per_sec', 0),
            time_to_first_token_ms=response.get('time_to_first_token_ms'),
            error=response.get('error')
        )

    def _run_warmup(self, system_prompt: str) -> None:
        """Run warmup queries to ensure model is loaded and ready."""
        warmup_prompt = "What is the IP address of pve-main?"

        for i in range(self.config.warmup_queries):
            try:
                self.engine.query(
                    prompt=warmup_prompt,
                    system_prompt=system_prompt,
                    max_tokens=50,
                    temperature=0.1
                )
            except Exception as e:
                print(f"[WARN] Warmup query {i+1} failed: {e}")

    def _generate_test_id(self) -> str:
        """Generate a unique test ID."""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_safe = self.config.model.replace(':', '-').replace('/', '-')
        return f"{self.config.gpu}-{self.config.engine}-{model_safe}-{timestamp}"

    def _detect_quantization(self) -> Optional[str]:
        """Attempt to detect model quantization from name."""
        model = self.config.model.lower()
        quant_patterns = [
            'q4_k_m', 'q4_k_s', 'q4_0', 'q4_1',
            'q5_k_m', 'q5_k_s', 'q5_0', 'q5_1',
            'q6_k', 'q8_0', 'fp16', 'awq', 'gptq'
        ]
        for pattern in quant_patterns:
            if pattern in model:
                return pattern.upper()
        return None

    def _get_gpu_count(self) -> int:
        """Get GPU count from configuration name."""
        gpu = self.config.gpu.lower()
        if '-4x' in gpu:
            return 4
        elif '-2x' in gpu:
            return 2
        elif '-3060ti' in gpu or '-3070' in gpu:
            return 2  # Mixed config
        return 1


def run_single_test(
    config: TestConfig,
    engine_adapter: Any,
    questions: List[Question],
    inventory: Dict
) -> Dict:
    """
    Convenience function to run a single test.

    Args:
        config: Test configuration
        engine_adapter: Engine adapter instance
        questions: List of questions
        inventory: Inventory data

    Returns:
        Test results summary
    """
    # Check for completed questions
    completed = set()
    if config.results_dir:
        completed = get_completed_question_ids(config.results_dir, config.model)

    if completed:
        print(f"Found {len(completed)} previously completed questions")

    runner = TestRunner(config, engine_adapter)
    return runner.run(questions, inventory, completed)


def load_models_from_config(filepath: str) -> List[str]:
    """
    Load model names from a config file.

    Config format: one model per line, # for comments.

    Args:
        filepath: Path to config file

    Returns:
        List of model names
    """
    models = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            models.append(line)

    return models


def get_model_status(
    results_dir: str,
    model_name: str,
    total_questions: int = 42
) -> Dict:
    """
    Get the status of a model's test run.

    Args:
        results_dir: Results directory
        model_name: Model name
        total_questions: Total number of questions

    Returns:
        Dictionary with status information
    """
    from .results import load_results, ResultsWriter

    sanitized = ResultsWriter._sanitize_name(model_name)
    model_dir = f"{results_dir}/{sanitized}"
    questions_file = f"{model_dir}/questions.jsonl"

    import os
    if not os.path.exists(questions_file):
        return {
            'status': 'pending',
            'symbol': '',
            'completed': 0,
            'total': total_questions,
            'avg_latency_ms': None,
            'avg_tokens_per_sec': None,
            'last_updated': None
        }

    results = load_results(questions_file)

    if not results:
        return {
            'status': 'pending',
            'symbol': '',
            'completed': 0,
            'total': total_questions,
            'avg_latency_ms': None,
            'avg_tokens_per_sec': None,
            'last_updated': None
        }

    completed = len(results)
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    tps = [r.tokens_per_sec for r in results if r.tokens_per_sec > 0]

    status = 'complete' if completed >= total_questions else 'partial'
    symbol = '' if status == 'complete' else ''

    last_timestamp = max(r.timestamp for r in results) if results else None

    return {
        'status': status,
        'symbol': symbol,
        'completed': completed,
        'total': total_questions,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else None,
        'avg_tokens_per_sec': sum(tps) / len(tps) if tps else None,
        'last_updated': last_timestamp
    }


def print_batch_status(
    results_dir: str,
    models: List[str],
    total_questions: int = 42
) -> None:
    """
    Print status table for batch testing.

    Args:
        results_dir: Results directory
        models: List of model names
        total_questions: Total questions per model
    """
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'Status':<12} {'Progress':<10} {'Latency':<12} {'Tok/s':<10} {'Updated':<20}")
    print("-" * 90)

    for model in models:
        status = get_model_status(results_dir, model, total_questions)

        latency_str = '-'
        if status['avg_latency_ms']:
            latency_str = f"{status['avg_latency_ms']:.0f}ms"

        tps_str = '-'
        if status['avg_tokens_per_sec']:
            tps_str = f"{status['avg_tokens_per_sec']:.1f}"

        updated_str = '-'
        if status['last_updated']:
            updated_str = status['last_updated'][:16].replace('T', ' ')

        status_display = f"{status['symbol']} {status['status'].capitalize()}"
        progress = f"{status['completed']}/{status['total']}"

        print(f"{model:<30} {status_display:<12} {progress:<10} {latency_str:<12} {tps_str:<10} {updated_str:<20}")

    print("=" * 90)
