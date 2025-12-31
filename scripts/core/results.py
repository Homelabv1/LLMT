"""
Results schema and JSONL writer for the LLM Testing Framework.

This module handles saving and loading test results, including atomic file
operations for crash safety and JSONL format for incremental writes.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Set, Union


class QuestionResult:
    """Represents the result of a single question test."""

    def __init__(
        self,
        q_id: int,
        category: str,
        question: str,
        expected: str,
        response: str,
        latency_ms: float,
        tokens_prompt: int,
        tokens_generated: int,
        tokens_per_sec: float,
        time_to_first_token_ms: Optional[float] = None,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
        score: Optional[int] = None,
        scorer_notes: Optional[str] = None
    ):
        """
        Initialize a QuestionResult instance.

        Args:
            q_id: Question identifier
            category: Question category
            question: The question text
            expected: Expected answer
            response: Model's response
            latency_ms: Total latency in milliseconds
            tokens_prompt: Number of prompt tokens
            tokens_generated: Number of generated tokens
            tokens_per_sec: Generation speed in tokens per second
            time_to_first_token_ms: Time to first token in milliseconds
            timestamp: ISO format timestamp
            error: Error message if query failed
            score: Manual score (0-3), None if not yet scored
            scorer_notes: Optional notes from scorer
        """
        self.q_id = q_id
        self.category = category
        self.question = question
        self.expected = expected
        self.response = response
        self.latency_ms = latency_ms
        self.tokens_prompt = tokens_prompt
        self.tokens_generated = tokens_generated
        self.tokens_per_sec = tokens_per_sec
        self.time_to_first_token_ms = time_to_first_token_ms
        self.timestamp = timestamp or datetime.now().isoformat()
        self.error = error
        self.score = score
        self.scorer_notes = scorer_notes

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'q_id': self.q_id,
            'category': self.category,
            'question': self.question,
            'expected': self.expected,
            'response': self.response,
            'latency_ms': self.latency_ms,
            'tokens_prompt': self.tokens_prompt,
            'tokens_generated': self.tokens_generated,
            'tokens_per_sec': self.tokens_per_sec,
            'time_to_first_token_ms': self.time_to_first_token_ms,
            'timestamp': self.timestamp,
            'error': self.error,
            'score': self.score,
            'scorer_notes': self.scorer_notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'QuestionResult':
        """Create a QuestionResult from a dictionary."""
        return cls(
            q_id=data['q_id'],
            category=data['category'],
            question=data['question'],
            expected=data['expected'],
            response=data.get('response', ''),
            latency_ms=data.get('latency_ms', 0),
            tokens_prompt=data.get('tokens_prompt', 0),
            tokens_generated=data.get('tokens_generated', 0),
            tokens_per_sec=data.get('tokens_per_sec', 0),
            time_to_first_token_ms=data.get('time_to_first_token_ms'),
            timestamp=data.get('timestamp'),
            error=data.get('error'),
            score=data.get('score'),
            scorer_notes=data.get('scorer_notes')
        )


class RunMetadata:
    """Metadata for a test run."""

    def __init__(
        self,
        test_id: str,
        timestamp: str,
        model_name: str,
        model_quantization: Optional[str],
        engine_name: str,
        engine_version: str,
        gpu_name: str,
        gpu_model: str,
        gpu_vram_mb: int,
        gpu_count: int = 1,
        test_config: Optional[Dict] = None,
        power_limits_applied: Optional[str] = None,
        total_questions: int = 100
    ):
        """
        Initialize RunMetadata instance.

        Args:
            test_id: Unique test run identifier
            timestamp: ISO format timestamp
            model_name: Model name (e.g., 'qwen3:4b')
            model_quantization: Quantization format if known
            engine_name: Inference engine name
            engine_version: Engine version
            gpu_name: GPU configuration name (e.g., '1660super-1x')
            gpu_model: GPU model name (e.g., 'GTX 1660 Super')
            gpu_vram_mb: Total VRAM in MB
            gpu_count: Number of GPUs
            test_config: Test configuration dictionary
            power_limits_applied: Power limits applied (e.g., '280,160')
            total_questions: Total number of questions
        """
        self.test_id = test_id
        self.timestamp = timestamp
        self.model_name = model_name
        self.model_quantization = model_quantization
        self.engine_name = engine_name
        self.engine_version = engine_version
        self.gpu_name = gpu_name
        self.gpu_model = gpu_model
        self.gpu_vram_mb = gpu_vram_mb
        self.gpu_count = gpu_count
        self.test_config = test_config or {}
        self.power_limits_applied = power_limits_applied
        self.total_questions = total_questions
        self.scored = 0
        self.total_score = None  # type: Optional[int]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        max_score = self.total_questions * 3
        percentage = None
        if self.total_score is not None:
            percentage = round(self.total_score / max_score * 100, 1)

        return {
            'test_id': self.test_id,
            'timestamp': self.timestamp,
            'model': {
                'name': self.model_name,
                'quantization': self.model_quantization
            },
            'engine': {
                'name': self.engine_name,
                'version': self.engine_version
            },
            'gpu': {
                'name': self.gpu_name,
                'model': self.gpu_model,
                'vram_total_mb': self.gpu_vram_mb,
                'count': self.gpu_count
            },
            'test_config': self.test_config,
            'power_limits_applied': self.power_limits_applied,
            'scoring': {
                'total_questions': self.total_questions,
                'scored': self.scored,
                'total_score': self.total_score,
                'max_score': max_score,
                'percentage': percentage
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RunMetadata':
        """Create RunMetadata from a dictionary."""
        meta = cls(
            test_id=data['test_id'],
            timestamp=data['timestamp'],
            model_name=data['model']['name'],
            model_quantization=data['model'].get('quantization'),
            engine_name=data['engine']['name'],
            engine_version=data['engine']['version'],
            gpu_name=data['gpu']['name'],
            gpu_model=data['gpu']['model'],
            gpu_vram_mb=data['gpu']['vram_total_mb'],
            gpu_count=data['gpu'].get('count', 1),
            test_config=data.get('test_config', {}),
            power_limits_applied=data.get('power_limits_applied'),
            total_questions=data['scoring']['total_questions']
        )
        meta.scored = data['scoring'].get('scored', 0)
        meta.total_score = data['scoring'].get('total_score')
        return meta


class ResultsWriter:
    """
    Handles writing test results with atomic operations for crash safety.

    Results are stored in JSONL format (one JSON object per line) to allow
    incremental writing and resumability after interruptions.
    """

    def __init__(self, results_dir: str, model_name: str):
        """
        Initialize ResultsWriter.

        Args:
            results_dir: Base directory for results
            model_name: Model name (used to create subdirectory)
        """
        # Sanitize model name for directory use
        self.model_dir_name = self._sanitize_name(model_name)
        self.model_dir = os.path.join(results_dir, self.model_dir_name)

        # Create directory if needed
        os.makedirs(self.model_dir, exist_ok=True)

        # File paths
        self.questions_file = os.path.join(self.model_dir, 'questions.jsonl')
        self.meta_file = os.path.join(self.model_dir, 'run_meta.json')
        self.metrics_file = os.path.join(self.model_dir, 'gpu_metrics.csv')
        self.summary_file = os.path.join(self.model_dir, 'gpu_summary.json')

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a name for use as a directory name."""
        # Replace colons, slashes, and other problematic characters
        return name.replace(':', '-').replace('/', '-').replace('\\', '-')

    def write_result(self, result: QuestionResult) -> None:
        """
        Atomically append a question result to the JSONL file.

        Args:
            result: QuestionResult to write
        """
        self._safe_append_jsonl(self.questions_file, result.to_dict())

    def write_metadata(self, metadata: RunMetadata) -> None:
        """
        Write or update run metadata.

        Args:
            metadata: RunMetadata to write
        """
        self._safe_write_json(self.meta_file, metadata.to_dict())

    def write_gpu_metrics_header(self) -> None:
        """Write the CSV header for GPU metrics."""
        header = 'timestamp,vram_used_mb,vram_total_mb,gpu_util_pct,temp_c,phase\n'
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            f.write(header)

    def append_gpu_metrics(
        self,
        timestamp: str,
        vram_used_mb: int,
        vram_total_mb: int,
        gpu_util_pct: int,
        temp_c: int,
        phase: str
    ) -> None:
        """
        Append a GPU metrics row.

        Args:
            timestamp: ISO format timestamp
            vram_used_mb: VRAM used in MB
            vram_total_mb: Total VRAM in MB
            gpu_util_pct: GPU utilization percentage
            temp_c: Temperature in Celsius
            phase: Current phase (idle, model_loading, inference)
        """
        line = f'{timestamp},{vram_used_mb},{vram_total_mb},{gpu_util_pct},{temp_c},{phase}\n'
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()

    def write_gpu_summary(self, summary: Dict) -> None:
        """
        Write GPU summary statistics.

        Args:
            summary: Dictionary with GPU summary data
        """
        self._safe_write_json(self.summary_file, summary)

    def _safe_append_jsonl(self, filepath: str, data: Dict) -> None:
        """
        Atomically append a line to a JSONL file.

        Uses a temp file and atomic replace to prevent corruption.

        Args:
            filepath: Path to the JSONL file
            data: Dictionary to append as JSON line
        """
        line = json.dumps(data) + '\n'

        # Read existing content
        existing = ''
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                existing = f.read()

        # Write to temp file in the same directory (for atomic replace)
        dir_path = os.path.dirname(filepath) or '.'
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(existing + line)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace
            os.replace(temp_path, filepath)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _safe_write_json(self, filepath: str, data: Dict) -> None:
        """
        Atomically write a JSON file.

        Args:
            filepath: Path to the JSON file
            data: Dictionary to write as JSON
        """
        dir_path = os.path.dirname(filepath) or '.'
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def update_result_score(
        self,
        q_id: int,
        score: int,
        scorer_notes: Optional[str] = None
    ) -> bool:
        """
        Update the score for a specific question result.

        Args:
            q_id: Question ID to update
            score: Score (0-3)
            scorer_notes: Optional notes from scorer

        Returns:
            True if update was successful, False if question not found
        """
        if not os.path.exists(self.questions_file):
            return False

        # Load all results
        results = load_results(self.questions_file)

        # Find and update the result
        found = False
        for result in results:
            if result.q_id == q_id:
                result.score = score
                result.scorer_notes = scorer_notes
                found = True
                break

        if not found:
            return False

        # Rewrite the file with updated results
        lines = [json.dumps(r.to_dict()) + '\n' for r in results]
        dir_path = os.path.dirname(self.questions_file) or '.'
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.writelines(lines)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.questions_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        return True


def load_results(filepath: str) -> List[QuestionResult]:
    """
    Load question results from a JSONL file.

    Handles malformed lines gracefully with warnings.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of QuestionResult objects
    """
    results = []

    if not os.path.exists(filepath):
        return results

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(QuestionResult.from_dict(data))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_num} in {filepath}: {e}")
            except KeyError as e:
                print(f"[WARN] Skipping line {line_num} with missing key {e}")

    return results


def get_completed_question_ids(results_dir: str, model_name: str) -> Set[int]:
    """
    Get the set of completed question IDs for a model.

    Args:
        results_dir: Base results directory
        model_name: Model name

    Returns:
        Set of completed question IDs
    """
    sanitized = ResultsWriter._sanitize_name(model_name)
    questions_file = os.path.join(results_dir, sanitized, 'questions.jsonl')

    if not os.path.exists(questions_file):
        return set()

    results = load_results(questions_file)
    return {r.q_id for r in results}


def load_run_metadata(results_dir: str, model_name: str) -> Optional[RunMetadata]:
    """
    Load run metadata for a model.

    Args:
        results_dir: Base results directory
        model_name: Model name

    Returns:
        RunMetadata if found, None otherwise
    """
    sanitized = ResultsWriter._sanitize_name(model_name)
    meta_file = os.path.join(results_dir, sanitized, 'run_meta.json')

    if not os.path.exists(meta_file):
        return None

    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return RunMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] Could not load metadata: {e}")
        return None


def append_failure_log(
    log_file: str,
    timestamp: str,
    model: str,
    engine: str,
    gpu: str,
    error_type: str,
    error_message: str
) -> None:
    """
    Append a failure entry to the CSV failure log.

    Args:
        log_file: Path to failures.log
        timestamp: ISO format timestamp
        model: Model name
        engine: Engine name
        gpu: GPU configuration name
        error_type: Type of error (e.g., 'download_timeout', 'load_failure')
        error_message: Error message
    """
    # Ensure file has header
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('timestamp,model,engine,gpu,error_type,error_message\n')

    # Escape any commas in the error message
    error_message = error_message.replace('"', '""')
    if ',' in error_message or '\n' in error_message:
        error_message = f'"{error_message}"'

    line = f'{timestamp},{model},{engine},{gpu},{error_type},{error_message}\n'
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line)
        f.flush()


def write_load_failure(
    results_dir: str,
    model_name: str,
    error_type: str,
    error_message: str,
    details: Optional[Dict] = None
) -> None:
    """
    Write a load failure JSON file for a model.

    Args:
        results_dir: Base results directory
        model_name: Model name
        error_type: Type of error
        error_message: Error message
        details: Optional additional details
    """
    sanitized = ResultsWriter._sanitize_name(model_name)
    model_dir = os.path.join(results_dir, sanitized)
    os.makedirs(model_dir, exist_ok=True)

    failure_file = os.path.join(model_dir, 'load_failure.json')

    data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'error_type': error_type,
        'error_message': error_message,
        'details': details or {}
    }

    with open(failure_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
