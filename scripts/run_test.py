#!/usr/bin/env python3
"""
Single model test CLI for the LLM Testing Framework.

Usage:
    python run_test.py --engine ollama --model qwen3:4b [options]

This script tests a single model against the homelab inventory questions
and saves results for later analysis.
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.questions import load_questions, load_inventory, create_system_prompt
from core.results import get_completed_question_ids, append_failure_log, write_load_failure
from core.runner import TestConfig, TestRunner
from core.metrics import get_gpu_info, get_gpu_metrics
from engines import get_adapter, DEFAULT_URLS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run LLM tests on a single model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with Ollama
    python run_test.py -e ollama -m qwen3:4b

    # Test with custom results directory
    python run_test.py -e ollama -m qwen3:4b -r results/test1

    # Test with custom timeouts
    python run_test.py -e ollama -m mistral:7b --download-timeout 3600 --load-timeout 900
        """
    )

    # Required arguments
    parser.add_argument(
        '--engine', '-e',
        required=True,
        choices=['ollama', 'llamacpp', 'vllm'],
        help='Inference engine to use'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Model name (e.g., qwen3:4b, /models/model.gguf)'
    )

    # Optional arguments
    parser.add_argument(
        '--gpu', '-g',
        default='1660super-1x',
        help='GPU identifier for results organization (default: 1660super-1x)'
    )
    parser.add_argument(
        '--url', '-u',
        help='Override server URL (default: engine-specific)'
    )
    parser.add_argument(
        '--questions', '-q',
        default=os.path.join(SCRIPT_DIR, 'data', 'homelab_test_questions.json'),
        help='Path to questions JSON file'
    )
    parser.add_argument(
        '--inventory', '-i',
        default=os.path.join(SCRIPT_DIR, 'data', 'homelab_inventory.json'),
        help='Path to inventory JSON file'
    )
    parser.add_argument(
        '--results-dir', '-r',
        help='Base results directory (default: gpus/nvidia/<gpu>/results/<engine>/)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=1,
        help='Number of warmup queries (default: 1)'
    )
    parser.add_argument(
        '--download-timeout',
        type=int,
        default=1800,
        help='Model download timeout in seconds (default: 1800 / 30 min)'
    )
    parser.add_argument(
        '--load-timeout',
        type=int,
        default=600,
        help='Model load timeout in seconds (default: 600 / 10 min)'
    )
    parser.add_argument(
        '--system-prompt', '-s',
        help='System prompt override (default: built-in template)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignoring any previous progress'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running tests'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Default: gpus/nvidia/<gpu>/results/<engine>/
        base_dir = os.path.dirname(SCRIPT_DIR)  # llm-testing/
        results_dir = os.path.join(
            base_dir, 'gpus', 'nvidia', args.gpu, 'results', args.engine
        )

    # Print header
    print("=" * 70)
    print("SINGLE MODEL TEST")
    print("=" * 70)
    print(f"Engine:           {args.engine}")
    print(f"Model:            {args.model}")
    print(f"GPU:              {args.gpu}")
    print(f"Results Dir:      {results_dir}")
    print(f"Download Timeout: {args.download_timeout}s ({args.download_timeout // 60} min)")
    print(f"Load Timeout:     {args.load_timeout}s ({args.load_timeout // 60} min)")
    print(f"Max Tokens:       {args.max_tokens}")
    print(f"Temperature:      {args.temperature}")
    print("=" * 70)

    # Get GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\nGPU: {gpu_info['name']}")
        print(f"VRAM: {gpu_info['vram_total_mb']}MB")
        print(f"Driver: {gpu_info['driver_version']}")
    else:
        print("\n[WARN] Could not detect GPU info via nvidia-smi")

    # Get current GPU metrics
    metrics = get_gpu_metrics()
    if metrics:
        print(f"Current: {metrics.vram_used_mb}MB VRAM, {metrics.temp_c}C")

    if args.dry_run:
        print("\n[DRY RUN] Would run tests with above configuration")
        return 0

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load questions and inventory
    print(f"\nLoading questions from: {args.questions}")
    try:
        questions = load_questions(args.questions)
        print(f"Loaded {len(questions)} questions")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"Loading inventory from: {args.inventory}")
    try:
        inventory = load_inventory(args.inventory)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    # Check for completed questions
    if args.no_resume:
        completed = set()
        print("\n[INFO] Starting fresh (--no-resume)")
    else:
        completed = get_completed_question_ids(results_dir, args.model)
        if completed:
            print(f"\n[RESUME] Found {len(completed)} completed questions")

    remaining = len(questions) - len(completed)
    if remaining == 0:
        print("\n[COMPLETE] All questions already answered!")
        return 0

    print(f"Questions to run: {remaining}")

    # Get engine adapter
    url = args.url or DEFAULT_URLS.get(args.engine)
    print(f"\nConnecting to {args.engine} at {url}...")

    try:
        adapter = get_adapter(args.engine, url)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    # Check engine availability
    if not adapter.is_available():
        print(f"[ERROR] {args.engine} is not reachable at {url}")
        return 1

    print(f"Connected! Version: {adapter.get_version()}")

    # Load model
    print(f"\nLoading model: {args.model}")
    print(f"  Timeouts: download={args.download_timeout}s, load={args.load_timeout}s")

    load_result = adapter.load_model(
        args.model,
        download_timeout_sec=args.download_timeout,
        load_timeout_sec=args.load_timeout
    )

    if not load_result['success']:
        error_msg = load_result.get('error', 'Unknown error')
        print(f"[ERROR] Failed to load model: {error_msg}")

        # Log failure
        failures_log = os.path.join(os.path.dirname(results_dir), 'failures.log')
        append_failure_log(
            failures_log,
            datetime.now().isoformat(),
            args.model,
            args.engine,
            args.gpu,
            'load_failure',
            error_msg
        )
        write_load_failure(results_dir, args.model, 'load_failure', error_msg)
        return 1

    print(f"Model loaded successfully ({load_result['load_time_sec']:.1f}s)")

    # Create test configuration
    config = TestConfig(
        engine=args.engine,
        model=args.model,
        gpu=args.gpu,
        url=url,
        questions_file=args.questions,
        inventory_file=args.inventory,
        results_dir=results_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_queries=args.warmup,
        download_timeout=args.download_timeout,
        load_timeout=args.load_timeout,
        system_prompt=args.system_prompt
    )

    # Run tests
    print(f"\nStarting tests...")
    runner = TestRunner(config, adapter)

    try:
        summary = runner.run(questions, inventory, completed)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test interrupted by user")
        summary = {'status': 'interrupted'}

    # Unload model
    print("\nUnloading model...")
    adapter.unload_model()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Status:           {summary.get('status', 'unknown')}")
    print(f"Questions Run:    {summary.get('questions_run', 0)}")
    print(f"Questions Failed: {summary.get('questions_failed', 0)}")

    if summary.get('avg_latency_ms'):
        print(f"Avg Latency:      {summary['avg_latency_ms']:.0f}ms")
    if summary.get('avg_tokens_per_sec'):
        print(f"Avg Tokens/sec:   {summary['avg_tokens_per_sec']:.1f}")

    print("=" * 70)

    return 0 if summary.get('status') == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
