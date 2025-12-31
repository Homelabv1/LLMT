#!/usr/bin/env python3
"""
Pre-flight check CLI for the LLM Testing Framework.

Verifies model availability and engine connectivity before test execution,
catching configuration errors early.

Usage:
    # Check all models for an engine
    python preflight_check.py --engine ollama --config models.txt

    # Check specific models
    python preflight_check.py --engine ollama --models qwen3:4b llama3.2:3b

    # Full check including load test
    python preflight_check.py --engine ollama --config models.txt --full

    # Check all engines for a GPU config
    python preflight_check.py --all-engines --gpu-config gpus/nvidia/1660super-1x/

    # JSON output for scripting
    python preflight_check.py --engine ollama --config models.txt --json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.preflight import (
    PreflightResult,
    ModelStatus,
    OverallStatus,
    get_preflight_checker,
    run_preflight,
    detect_gpu,
)
from engines import DEFAULT_URLS


# =============================================================================
# Output Formatting
# =============================================================================

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def supports_color() -> bool:
    """Check if terminal supports color."""
    if os.environ.get('NO_COLOR'):
        return False
    if not hasattr(sys.stdout, 'isatty'):
        return False
    return sys.stdout.isatty()


USE_COLOR = supports_color()


def color(text: str, color_code: str) -> str:
    """Apply color to text if supported."""
    if USE_COLOR:
        return f"{color_code}{text}{Colors.RESET}"
    return text


def status_icon(success: bool, warning: bool = False) -> str:
    """Get status icon."""
    if success and not warning:
        return color("✓", Colors.GREEN)
    elif warning:
        return color("⚠", Colors.YELLOW)
    else:
        return color("✗", Colors.RED)


def print_header(title: str, char: str = "=", width: int = 70):
    """Print formatted header."""
    print(char * width)
    print(color(title, Colors.BOLD))
    print(char * width)


def print_section(title: str, number: Optional[int] = None, total: Optional[int] = None):
    """Print section header."""
    if number and total:
        print(f"\n[{number}/{total}] {color(title, Colors.BOLD)}")
    else:
        print(f"\n{color(title, Colors.BOLD)}")


def format_size(size_gb: Optional[float]) -> str:
    """Format size in GB."""
    if size_gb is None:
        return "unknown"
    if size_gb < 1:
        return f"{size_gb * 1024:.0f}MB"
    return f"{size_gb:.1f}GB"


def print_connectivity(result: PreflightResult):
    """Print connectivity check results."""
    print_section("Connectivity Check", 1, 2)

    conn = result.connectivity

    # Service status
    icon = status_icon(conn.service_running)
    print(f"  {icon} {result.engine.capitalize()} service {'running' if conn.service_running else 'not running'}")

    # API status
    if conn.service_running:
        icon = status_icon(conn.api_reachable)
        version_str = f" (v{conn.api_version})" if conn.api_version else ""
        print(f"  {icon} API endpoint {'reachable' if conn.api_reachable else 'not reachable'}{version_str}")

    # Error message
    if conn.error_message:
        print(f"    {color('Error:', Colors.RED)} {conn.error_message}")

    # GPU status
    if conn.gpu_detected:
        icon = status_icon(True)
        vram_str = f" ({conn.gpu_vram_gb:.0f}GB)" if conn.gpu_vram_gb else ""
        print(f"  {icon} GPU detected: {conn.gpu_name}{vram_str}")
    else:
        icon = status_icon(False, warning=True)
        print(f"  {icon} No GPU detected (will use CPU)")


def print_model_checks(result: PreflightResult):
    """Print model check results."""
    print_section("Model Availability", 2, 2)
    print(f"  Checking {len(result.models)} models...\n")

    for model in result.models:
        print(f"  {color(model.name, Colors.CYAN)}")

        # Status-specific output
        if model.status == ModelStatus.READY_CACHED:
            icon = status_icon(True)
            size_str = f" ({format_size(model.size_gb)})" if model.size_gb else ""
            print(f"    {icon} Cached locally{size_str}")

        elif model.status == ModelStatus.READY_NEEDS_DOWNLOAD:
            icon = status_icon(True)
            print(f"    {icon} Valid model")
            icon = status_icon(False, warning=True)
            size_str = f" ({format_size(model.size_gb)})" if model.size_gb else ""
            print(f"    {icon} Not cached - will download{size_str}")

        elif model.status == ModelStatus.WARNING_TIGHT_VRAM:
            icon = status_icon(True)
            size_str = f" ({format_size(model.size_gb)})" if model.size_gb else ""
            print(f"    {icon} Cached locally{size_str}")
            icon = status_icon(True, warning=True)
            print(f"    {icon} VRAM estimate: {format_size(model.vram_estimate_gb)} (tight fit)")

        elif model.status == ModelStatus.WARNING_NEEDS_DOWNLOAD:
            icon = status_icon(True)
            print(f"    {icon} Valid model")
            icon = status_icon(False, warning=True)
            size_str = f" ({format_size(model.size_gb)})" if model.size_gb else ""
            print(f"    {icon} Not cached - will download{size_str}")
            icon = status_icon(True, warning=True)
            print(f"    {icon} VRAM estimate: {format_size(model.vram_estimate_gb)} (tight fit)")

        elif model.status == ModelStatus.FAILED_NOT_FOUND:
            icon = status_icon(False)
            print(f"    {icon} Not found in registry")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        elif model.status == ModelStatus.FAILED_INVALID:
            icon = status_icon(False)
            print(f"    {icon} Invalid model name or format")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        elif model.status == ModelStatus.FAILED_NO_SPACE:
            icon = status_icon(False)
            print(f"    {icon} Insufficient VRAM")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        elif model.status == ModelStatus.FAILED_AUTH:
            icon = status_icon(False)
            print(f"    {icon} Authentication required")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        elif model.status == ModelStatus.FAILED_LOAD:
            icon = status_icon(False)
            print(f"    {icon} Failed to load")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        elif model.status == ModelStatus.FAILED_QUERY:
            icon = status_icon(False)
            print(f"    {icon} Load succeeded but query failed")
            if model.error_message:
                print(f"      {color('Error:', Colors.RED)} {model.error_message}")

        else:
            icon = status_icon(False, warning=True)
            print(f"    {icon} Status: {model.status.value}")

        # VRAM estimate (if not already shown)
        if model.vram_estimate_gb and model.status not in (
            ModelStatus.WARNING_TIGHT_VRAM,
            ModelStatus.WARNING_NEEDS_DOWNLOAD,
            ModelStatus.FAILED_NO_SPACE
        ):
            icon = status_icon(model.fits_vram, warning=(model.vram_fit_status == "tight_fit"))
            fit_note = ""
            if model.vram_fit_status == "tight_fit":
                fit_note = " (tight fit)"
            elif model.vram_fit_status == "comfortable":
                fit_note = f" (fits in {result.connectivity.gpu_vram_gb:.0f}GB)"
            print(f"    {icon} VRAM estimate: {format_size(model.vram_estimate_gb)}{fit_note}")

        print()


def print_summary(result: PreflightResult):
    """Print summary section."""
    print_header("SUMMARY")

    # Count by status
    ready_cached = sum(1 for m in result.models if m.status == ModelStatus.READY_CACHED)
    ready_download = sum(1 for m in result.models if m.status in (
        ModelStatus.READY_NEEDS_DOWNLOAD, ModelStatus.WARNING_NEEDS_DOWNLOAD
    ))
    warnings = sum(1 for m in result.models if m.status.value.startswith("warning"))
    failed = sum(1 for m in result.models if m.status.value.startswith("failed"))

    print(f"Total models: {len(result.models)}")
    icon = status_icon(True)
    print(f"  {icon} Ready (cached): {ready_cached}")
    if ready_download > 0:
        icon = status_icon(True, warning=True)
        print(f"  {icon} Ready (need download): {ready_download}")
    if warnings > 0:
        icon = status_icon(True, warning=True)
        print(f"  {icon} Warnings: {warnings}")
    if failed > 0:
        icon = status_icon(False)
        print(f"  {icon} Failed: {failed}")

    # Download estimate
    total_download = sum(m.size_gb or 0 for m in result.models if not m.cached and m.size_gb)
    if total_download > 0:
        print(f"\nEstimated download: {format_size(total_download)}")
        # Rough time estimate at 50 Mbps
        time_minutes = total_download / (50 / 8 / 1024) / 60
        print(f"Estimated time: ~{time_minutes:.0f} minutes (at 50Mbps)")

    # Disk space
    print()
    disk = result.disk
    if disk.required_gb > 0:
        icon = status_icon(disk.has_space)
        print(f"Disk space available: {format_size(disk.available_gb)} {icon}")
        print(f"Disk space required: {format_size(disk.required_gb)} {icon}")

    # Failed models
    failed_models = [m for m in result.models if m.status.value.startswith("failed")]
    if failed_models:
        print(f"\n{color('Failed models:', Colors.RED)}")
        for m in failed_models:
            reason = m.error_message or m.status.value
            print(f"  - {m.name} ({reason})")

    # Warnings
    if result.warnings:
        print(f"\n{color('Warnings:', Colors.YELLOW)}")
        for w in result.warnings:
            print(f"  - {w}")

    # Overall status
    print()
    if result.overall_status == OverallStatus.PASSED:
        print(color("✓ Pre-flight PASSED", Colors.GREEN))
    elif result.overall_status == OverallStatus.WARNING:
        print(color("⚠ Pre-flight PASSED with warnings", Colors.YELLOW))
    else:
        print(color("✗ Pre-flight FAILED", Colors.RED))

    if result.can_proceed:
        print("  Ready to run tests.")
    else:
        print("  Fix issues before running tests.")

    print("=" * 70)


def print_result(result: PreflightResult):
    """Print full pre-flight result."""
    print_header(f"PRE-FLIGHT CHECK: {result.engine.capitalize()}")
    print(f"Engine URL: {result.engine_url}")

    print_connectivity(result)
    print_model_checks(result)
    print_summary(result)


def print_json(result: PreflightResult):
    """Print result as JSON."""
    print(json.dumps(result.to_dict(), indent=2))


def print_dry_run(engine: str, models: List[str], url: str):
    """Print dry run information."""
    print_header("DRY RUN")
    print(f"Engine: {engine}")
    print(f"URL: {url}")
    print(f"\nWould check {len(models)} models:")
    for m in models:
        print(f"  - {m}")
    print("=" * 70)


# =============================================================================
# Model Loading
# =============================================================================

def load_models_from_file(path: str) -> List[str]:
    """Load model list from file."""
    models = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                models.append(line)
    return models


def find_config_files(gpu_config_dir: str) -> dict:
    """Find model config files for each engine in GPU config directory."""
    configs = {}
    gpu_path = Path(gpu_config_dir)

    for engine in ['ollama', 'llamacpp', 'vllm']:
        # Try different naming conventions
        patterns = [
            gpu_path / 'configs' / f'{engine}_models.txt',
            gpu_path / 'configs' / f'{engine}.txt',
            gpu_path / f'{engine}_models.txt',
        ]

        for pattern in patterns:
            if pattern.exists():
                configs[engine] = str(pattern)
                break

    return configs


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pre-flight check for LLM testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check all models for an engine
    python preflight_check.py --engine ollama --config models.txt

    # Check specific models
    python preflight_check.py --engine ollama --models qwen3:4b llama3.2:3b

    # Full check including load test
    python preflight_check.py --engine ollama --config models.txt --full

    # Check all engines
    python preflight_check.py --all-engines --gpu-config gpus/nvidia/1660super-1x/

    # JSON output
    python preflight_check.py --engine ollama --config models.txt --json
        """
    )

    # Engine selection
    engine_group = parser.add_mutually_exclusive_group(required=True)
    engine_group.add_argument(
        '--engine', '-e',
        choices=['ollama', 'llamacpp', 'vllm'],
        help='Engine to check'
    )
    engine_group.add_argument(
        '--all-engines',
        action='store_true',
        help='Check all engines'
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to model config file'
    )
    model_group.add_argument(
        '--models', '-m',
        nargs='+',
        help='Specific model names/tags to check'
    )
    model_group.add_argument(
        '--gpu-config', '-g',
        type=Path,
        help='GPU config directory (checks all engines)'
    )

    # Check options
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Include load test (loads model, sends test prompt)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be checked without checking'
    )

    # Output options
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show summary and errors'
    )

    # Engine options
    parser.add_argument(
        '--url', '-u',
        help='Override engine URL'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=30,
        help='Timeout for connectivity checks (seconds)'
    )

    # Skip options
    parser.add_argument(
        '--skip-disk-check',
        action='store_true',
        help='Skip disk space verification'
    )
    parser.add_argument(
        '--skip-vram-check',
        action='store_true',
        help='Skip VRAM estimation check'
    )

    # llama.cpp specific
    parser.add_argument(
        '--models-dir',
        default='/models',
        help='Models directory for llama.cpp (default: /models)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine engines to check
    if args.all_engines:
        engines = ['ollama', 'llamacpp', 'vllm']
    else:
        engines = [args.engine]

    # Determine models to check
    if args.gpu_config:
        # Find config files for each engine
        config_files = find_config_files(args.gpu_config)
        if not config_files:
            print(f"[ERROR] No config files found in {args.gpu_config}")
            return 1
    elif args.config:
        if not args.config.exists():
            print(f"[ERROR] Config file not found: {args.config}")
            return 1
        config_files = {engines[0]: str(args.config)}
    elif args.models:
        config_files = {engines[0]: None}  # Use --models directly
    else:
        print("[ERROR] Provide --config, --models, or --gpu-config")
        return 1

    # Run checks for each engine
    all_results = []
    any_failed = False

    for engine in engines:
        # Get models for this engine
        if engine in config_files:
            config_path = config_files.get(engine)
            if config_path:
                models = load_models_from_file(config_path)
            elif args.models:
                models = args.models
            else:
                continue
        else:
            continue

        if not models:
            if not args.json:
                print(f"[WARN] No models to check for {engine}")
            continue

        # Get URL
        url = args.url or DEFAULT_URLS.get(engine)

        # Dry run
        if args.dry_run:
            if not args.json:
                print_dry_run(engine, models, url)
            continue

        # Run check
        try:
            result = run_preflight(
                engine=engine,
                models=models,
                url=url,
                full=args.full,
                skip_disk_check=args.skip_disk_check,
                skip_vram_check=args.skip_vram_check,
                timeout=args.timeout,
                models_dir=args.models_dir,
            )
            all_results.append(result)

            if result.overall_status == OverallStatus.FAILED:
                any_failed = True

            # Output
            if args.json:
                print_json(result)
            else:
                print_result(result)
                print()

        except Exception as e:
            if args.json:
                print(json.dumps({"error": str(e), "engine": engine}))
            else:
                print(f"[ERROR] Check failed for {engine}: {e}")
            any_failed = True

    if args.dry_run:
        return 0

    return 1 if any_failed else 0


if __name__ == '__main__':
    sys.exit(main())
