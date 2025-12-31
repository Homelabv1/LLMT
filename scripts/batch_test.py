#!/usr/bin/env python3
"""
Batch test runner CLI for the LLM Testing Framework.

Usage:
    python batch_test.py --engine ollama --config models.txt [options]
    python batch_test.py --engine ollama --models qwen3:4b llama3.2:3b [options]
    python batch_test.py --status --results-dir results/

This script runs tests on multiple models sequentially with proper GPU
cooldown between models, resumability, and graceful shutdown handling.
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional, Set

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.questions import load_questions, load_inventory
from core.results import (
    get_completed_question_ids,
    append_failure_log,
    write_load_failure,
)
from core.runner import (
    TestConfig,
    TestRunner,
    load_models_from_config,
    print_batch_status,
    get_model_status,
)
from core.metrics import (
    get_gpu_info,
    get_gpu_metrics,
    wait_for_cooldown,
    apply_power_limits,
    verify_vram_cleared,
)
from engines import get_adapter, DEFAULT_URLS
from core.preflight import run_preflight, OverallStatus, ModelStatus


# Global shutdown flag
_shutdown_requested = False
_force_shutdown = False


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested, _force_shutdown

    if _shutdown_requested:
        print("\n[FORCE] Second interrupt - forcing exit...")
        _force_shutdown = True
        sys.exit(1)
    else:
        print("\n[SHUTDOWN] Will exit after current model completes...")
        print("[SHUTDOWN] Press Ctrl+C again to force immediate exit")
        _shutdown_requested = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run LLM tests on multiple models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test models from config file
    python batch_test.py -e ollama -c models.txt

    # Test specific models
    python batch_test.py -e ollama -m qwen3:4b llama3.2:3b phi4-mini:3.8b

    # Show status
    python batch_test.py -e ollama -c models.txt --status

    # With power limits
    python batch_test.py -e ollama -c models.txt --power-limit "280,160"
        """
    )

    # Engine selection
    parser.add_argument(
        '--engine', '-e',
        choices=['ollama', 'llamacpp', 'vllm'],
        help='Inference engine to use'
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--models', '-m',
        nargs='+',
        help='List of models to test'
    )
    model_group.add_argument(
        '--config', '-c',
        help='Path to model list file (one model per line)'
    )

    # GPU and results
    parser.add_argument(
        '--gpu', '-g',
        default='1660super-1x',
        help='GPU identifier (default: 1660super-1x)'
    )
    parser.add_argument(
        '--url', '-u',
        help='Override server URL'
    )
    parser.add_argument(
        '--results-dir', '-r',
        help='Results directory (default: gpus/nvidia/<gpu>/results/<engine>/)'
    )

    # Cooldown settings
    parser.add_argument(
        '--min-cooldown',
        type=int,
        default=30,
        help='Minimum cooldown time in seconds (default: 30)'
    )
    parser.add_argument(
        '--target-temp',
        type=int,
        default=50,
        help='Target GPU temperature in Celsius (default: 50)'
    )

    # Timeouts
    parser.add_argument(
        '--download-timeout',
        type=int,
        default=1800,
        help='Download timeout in seconds (default: 1800 / 30 min)'
    )
    parser.add_argument(
        '--load-timeout',
        type=int,
        default=600,
        help='Load timeout in seconds (default: 600 / 10 min)'
    )

    # Generation settings
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

    # Power management
    parser.add_argument(
        '--power-limit',
        help='Comma-separated power limits per GPU in watts (e.g., "280,160")'
    )
    parser.add_argument(
        '--power-threshold',
        type=int,
        help='Estimated system power threshold in watts'
    )
    parser.add_argument(
        '--skip-power-limit',
        action='store_true',
        help="Don't apply power limits even if specified"
    )

    # Data files
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

    # Actions
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show progress and exit without running tests'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview configuration without executing'
    )

    # Pre-flight checks
    parser.add_argument(
        '--preflight',
        action='store_true',
        help='Run pre-flight checks before testing'
    )
    parser.add_argument(
        '--preflight-full',
        action='store_true',
        help='Run full pre-flight including load tests'
    )
    parser.add_argument(
        '--preflight-only',
        action='store_true',
        help='Only run pre-flight, do not start tests'
    )
    parser.add_argument(
        '--skip-preflight-vram',
        action='store_true',
        help='Skip VRAM estimation in pre-flight'
    )

    return parser.parse_args()


def get_models(args) -> List[str]:
    """Get list of models from arguments."""
    if args.models:
        return args.models
    elif args.config:
        if not os.path.exists(args.config):
            print(f"[ERROR] Config file not found: {args.config}")
            sys.exit(1)
        return load_models_from_config(args.config)
    else:
        return []


def count_status(results_dir: str, models: List[str], total_questions: int = 100):
    """Count models by status."""
    complete = 0
    partial = 0
    pending = 0

    for model in models:
        status = get_model_status(results_dir, model, total_questions)
        if status['status'] == 'complete':
            complete += 1
        elif status['status'] == 'partial':
            partial += 1
        else:
            pending += 1

    return complete, partial, pending


def run_preflight_check(
    engine: str,
    models: List[str],
    url: str,
    full: bool = False,
    skip_vram: bool = False
) -> bool:
    """
    Run pre-flight checks and return True if can proceed.

    Args:
        engine: Engine name
        models: List of model names
        url: Engine URL
        full: Include load tests
        skip_vram: Skip VRAM checks

    Returns:
        True if pre-flight passed and tests can proceed
    """
    print("\n" + "=" * 70)
    print("PRE-FLIGHT CHECK")
    print("=" * 70)

    result = run_preflight(
        engine=engine,
        models=models,
        url=url,
        full=full,
        skip_vram_check=skip_vram,
        timeout=30
    )

    # Print connectivity status
    conn = result.connectivity
    if conn.service_running and conn.api_reachable:
        print(f"✓ {engine.capitalize()} service running (v{conn.api_version or 'unknown'})")
    else:
        print(f"✗ {engine.capitalize()} service not available")
        if conn.error_message:
            print(f"  Error: {conn.error_message}")

    if conn.gpu_detected:
        print(f"✓ GPU: {conn.gpu_name} ({conn.gpu_vram_gb:.0f}GB)")
    else:
        print("⚠ No GPU detected")

    # Count results
    ready_cached = sum(1 for m in result.models if m.status == ModelStatus.READY_CACHED)
    ready_download = sum(1 for m in result.models if m.status in (
        ModelStatus.READY_NEEDS_DOWNLOAD, ModelStatus.WARNING_NEEDS_DOWNLOAD
    ))
    warnings = sum(1 for m in result.models if m.status.value.startswith("warning"))
    failed = sum(1 for m in result.models if m.status.value.startswith("failed"))

    print(f"\nModels: {len(result.models)} total")
    print(f"  ✓ Ready (cached): {ready_cached}")
    if ready_download > 0:
        print(f"  ⚠ Ready (need download): {ready_download}")
    if warnings > 0:
        print(f"  ⚠ Warnings: {warnings}")
    if failed > 0:
        print(f"  ✗ Failed: {failed}")

    # Show failed models
    failed_models = [m for m in result.models if m.status.value.startswith("failed")]
    if failed_models:
        print("\nFailed models:")
        for m in failed_models:
            reason = m.error_message or m.status.value
            print(f"  - {m.name}: {reason}")

    # Show warnings
    warning_models = [m for m in result.models if m.status.value.startswith("warning")]
    if warning_models:
        print("\nWarnings:")
        for m in warning_models:
            print(f"  - {m.name}: {m.vram_fit_status}")

    print("=" * 70)

    # Determine if we can proceed
    if result.overall_status == OverallStatus.FAILED:
        print("✗ Pre-flight FAILED")
        if not result.can_proceed:
            print("  Cannot proceed with tests.")
            return False
        else:
            print("  Some models failed, but others can proceed.")
    elif result.overall_status == OverallStatus.WARNING:
        print("⚠ Pre-flight passed with warnings")
        # Ask for confirmation
        try:
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Aborted by user.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return False
    else:
        print("✓ Pre-flight PASSED")

    print()
    return True


def main():
    """Main entry point."""
    global _shutdown_requested

    args = parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get models list
    models = get_models(args)

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    elif args.engine and args.gpu:
        base_dir = os.path.dirname(SCRIPT_DIR)
        results_dir = os.path.join(
            base_dir, 'gpus', 'nvidia', args.gpu, 'results', args.engine
        )
    else:
        results_dir = '.'

    # Status-only mode
    if args.status:
        if not models:
            print("[ERROR] Provide --models or --config for status check")
            return 1

        # Load questions to get count
        try:
            questions = load_questions(args.questions)
            total_q = len(questions)
        except Exception:
            total_q = 100  # Default question count

        print_batch_status(results_dir, models, total_q)
        return 0

    # Validate required arguments for running tests
    if not args.engine:
        print("[ERROR] --engine is required for running tests")
        return 1

    if not models:
        print("[ERROR] Provide --models or --config")
        return 1

    # Get URL
    url = args.url or DEFAULT_URLS.get(args.engine)

    # Pre-flight check
    if args.preflight or args.preflight_full or args.preflight_only:
        can_proceed = run_preflight_check(
            engine=args.engine,
            models=models,
            url=url,
            full=args.preflight_full,
            skip_vram=args.skip_preflight_vram
        )

        if args.preflight_only:
            if can_proceed:
                print("Pre-flight passed. Exiting (--preflight-only)")
                return 0
            else:
                return 1

        if not can_proceed:
            return 1

        print("Pre-flight passed. Starting tests...\n")

    # Print header
    print("=" * 70)
    print("BATCH TEST RUNNER")
    print("=" * 70)
    print(f"Engine:           {args.engine}")
    print(f"GPU:              {args.gpu}")
    print(f"Models:           {len(models)}")
    print(f"Min cooldown:     {args.min_cooldown}s")
    print(f"Target temp:      {args.target_temp}C")
    print(f"Download timeout: {args.download_timeout}s ({args.download_timeout // 60} min)")
    print(f"Load timeout:     {args.load_timeout}s ({args.load_timeout // 60} min)")
    if args.power_limit:
        print(f"Power limits:     {args.power_limit}W")
    if args.power_threshold:
        print(f"Power threshold:  {args.power_threshold}W")
    print("=" * 70)

    # Get initial GPU state
    gpu_info = get_gpu_info()
    metrics = get_gpu_metrics()

    if gpu_info:
        print(f"\nGPU: {gpu_info['name']} ({gpu_info['vram_total_mb']}MB)")
    if metrics:
        print(f"Initial state: {metrics.vram_used_mb}MB VRAM, {metrics.temp_c}C")
    baseline_vram = metrics.vram_used_mb if metrics else 500

    if args.dry_run:
        print(f"\n[DRY RUN] Would test {len(models)} models:")
        for m in models:
            print(f"  - {m}")
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

    # Get engine adapter
    print(f"\nConnecting to {args.engine} at {url}...")

    try:
        adapter = get_adapter(args.engine, url)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 1

    if not adapter.is_available():
        print(f"[ERROR] {args.engine} is not reachable at {url}")
        return 1

    print(f"Connected! Version: {adapter.get_version()}")

    # Apply power limits
    if args.power_limit and not args.skip_power_limit:
        print(f"\nApplying power limits: {args.power_limit}W")
        apply_power_limits(args.power_limit)

    # Show initial status
    complete, partial, pending = count_status(results_dir, models, len(questions))
    print(f"\nStatus:")
    print(f"  Complete: {complete}")
    print(f"  Partial:  {partial}")
    print(f"  Pending:  {pending}")

    # Failures log
    failures_log = os.path.join(results_dir, '..', 'failures.log')

    # Run tests for each model
    successful = 0
    failed = 0

    for i, model in enumerate(models, 1):
        if _shutdown_requested:
            print(f"\n[SHUTDOWN] Stopping before model {i}/{len(models)}")
            break

        # Check if model is already complete
        completed = get_completed_question_ids(results_dir, model)
        if len(completed) >= len(questions):
            print(f"\n[SKIP] {model}: Already complete ({len(completed)}/{len(questions)})")
            successful += 1
            continue

        print("\n" + "=" * 70)
        print(f"MODEL {i}/{len(models)}: {model}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Get pre-load GPU state
        metrics = get_gpu_metrics()
        if metrics:
            print(f"  Pre-load GPU: {metrics.vram_used_mb}MB VRAM, {metrics.temp_c}C")

        # Load model
        print(f"Loading model: {model}")
        print(f"  Timeouts: download={args.download_timeout}s, load={args.load_timeout}s")

        load_result = adapter.load_model(
            model,
            download_timeout_sec=args.download_timeout,
            load_timeout_sec=args.load_timeout
        )

        if not load_result['success']:
            error_msg = load_result.get('error', 'Unknown error')
            print(f"[ERROR] Failed to load model: {error_msg}")

            append_failure_log(
                failures_log,
                datetime.now().isoformat(),
                model,
                args.engine,
                args.gpu,
                'load_failure',
                error_msg
            )
            write_load_failure(results_dir, model, 'load_failure', error_msg)
            failed += 1
            continue

        if load_result.get('downloaded'):
            print(f"  (Model was downloaded)")
        else:
            print(f"  (Model was already downloaded)")

        # Show remaining questions
        remaining = len(questions) - len(completed)
        print(f"Questions: {len(completed)} completed, {remaining} remaining")

        # Create test config
        config = TestConfig(
            engine=args.engine,
            model=model,
            gpu=args.gpu,
            url=url,
            results_dir=results_dir,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            warmup_queries=1 if len(completed) == 0 else 0,
            download_timeout=args.download_timeout,
            load_timeout=args.load_timeout,
            min_cooldown=args.min_cooldown,
            target_temp=args.target_temp,
            power_limit=args.power_limit,
            power_threshold=args.power_threshold
        )

        # Run tests
        runner = TestRunner(config, adapter)
        try:
            summary = runner.run(questions, inventory, completed)
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            summary = {'status': 'error', 'error': str(e)}

        # Report result
        if summary.get('status') == 'complete':
            print(f"\n{model} completed successfully")
            successful += 1
        elif summary.get('status') == 'interrupted':
            print(f"\n{model} interrupted")
            _shutdown_requested = True
        elif summary.get('status') == 'power_aborted':
            print(f"\n{model} aborted due to power threshold")
            failed += 1
        else:
            print(f"\n{model} incomplete: {summary.get('status', 'unknown')}")
            failed += 1

        # Unload model
        print("\n  Clearing GPU memory...")
        metrics_before = get_gpu_metrics()
        if metrics_before:
            print(f"    VRAM before unload: {metrics_before.vram_used_mb}MB")

        adapter.unload_model()
        time.sleep(2)

        metrics_after = get_gpu_metrics()
        if metrics_after:
            print(f"    VRAM after unload: {metrics_after.vram_used_mb}MB")

        if verify_vram_cleared(baseline_vram, tolerance_pct=0.3, timeout_sec=30):
            print(f"    GPU memory cleared")
        else:
            print(f"    [WARN] GPU memory may not be fully cleared")

        # Post-unload status
        metrics = get_gpu_metrics()
        if metrics:
            print(f"  Post-unload GPU: {metrics.vram_used_mb}MB VRAM, {metrics.temp_c}C")

        # Cooldown before next model (unless last model or shutdown)
        if i < len(models) and not _shutdown_requested:
            print(f"\n  Cooldown between models...")
            wait_for_cooldown(
                target_temp_c=args.target_temp,
                min_wait_sec=args.min_cooldown,
                max_wait_sec=300,
                baseline_vram_mb=baseline_vram,
                verbose=True
            )

    # Print final summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total Models:     {len(models)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Skipped:          {len(models) - successful - failed}")

    if _shutdown_requested:
        print(f"Status:           Interrupted by user")
    else:
        print(f"Status:           Complete")

    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
