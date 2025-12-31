#!/usr/bin/env python3
"""
Results analysis CLI for the LLM Testing Framework.

Usage:
    python analyze_results.py results/ollama/
    python analyze_results.py results/ --compare
    python analyze_results.py results/ollama/qwen3-4b/ --detail

Analyzes test results and provides summaries, comparisons, and exports.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.results import load_results, QuestionResult


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze LLM test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single model results
    python analyze_results.py results/ollama/qwen3-4b/

    # Compare all models in a directory
    python analyze_results.py results/ollama/ --compare

    # Export comparison to CSV
    python analyze_results.py results/ollama/ --compare --export comparison.csv

    # Detailed breakdown
    python analyze_results.py results/ollama/qwen3-4b/ --detail
        """
    )

    parser.add_argument(
        'results_dir',
        help='Results directory to analyze'
    )
    parser.add_argument(
        '--detail',
        action='store_true',
        help='Show detailed breakdown for single run'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple runs'
    )
    parser.add_argument(
        '--export',
        metavar='FILE',
        help='Export comparison to CSV file'
    )
    parser.add_argument(
        '--by-category',
        action='store_true',
        help='Break down results by question category'
    )
    parser.add_argument(
        '--by-difficulty',
        action='store_true',
        help='Break down results by difficulty'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    return parser.parse_args()


def load_run_meta(model_dir: str) -> Optional[Dict]:
    """Load run metadata for a model."""
    meta_file = os.path.join(model_dir, 'run_meta.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            return json.load(f)
    return None


def analyze_single_run(model_dir: str) -> Dict:
    """Analyze results for a single model run."""
    questions_file = os.path.join(model_dir, 'questions.jsonl')

    if not os.path.exists(questions_file):
        return {'error': 'No results found'}

    results = load_results(questions_file)

    if not results:
        return {'error': 'No results to analyze'}

    # Basic stats
    total = len(results)
    errors = sum(1 for r in results if r.error)
    successful = total - errors

    # Latency stats
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    tps_values = [r.tokens_per_sec for r in results if r.tokens_per_sec > 0]
    ttft_values = [r.time_to_first_token_ms for r in results if r.time_to_first_token_ms]

    # Scoring stats
    scored = [r for r in results if r.score is not None]
    scores = [r.score for r in scored]

    analysis = {
        'total_questions': total,
        'successful': successful,
        'errors': errors,
        'error_rate': round(errors / total * 100, 1) if total > 0 else 0,
    }

    if latencies:
        analysis['latency_ms'] = {
            'min': round(min(latencies), 1),
            'max': round(max(latencies), 1),
            'avg': round(sum(latencies) / len(latencies), 1),
            'p50': round(sorted(latencies)[len(latencies) // 2], 1),
            'p95': round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) > 20 else None
        }

    if tps_values:
        analysis['tokens_per_sec'] = {
            'min': round(min(tps_values), 1),
            'max': round(max(tps_values), 1),
            'avg': round(sum(tps_values) / len(tps_values), 1)
        }

    if ttft_values:
        analysis['time_to_first_token_ms'] = {
            'min': round(min(ttft_values), 1),
            'max': round(max(ttft_values), 1),
            'avg': round(sum(ttft_values) / len(ttft_values), 1)
        }

    if scores:
        analysis['scoring'] = {
            'scored': len(scored),
            'unscored': total - len(scored),
            'total_score': sum(scores),
            'max_possible': total * 3,
            'percentage': round(sum(scores) / (total * 3) * 100, 1) if total > 0 else 0,
            'avg_score': round(sum(scores) / len(scores), 2)
        }

    # By category
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {'total': 0, 'errors': 0, 'latencies': [], 'scores': []}
        by_category[r.category]['total'] += 1
        if r.error:
            by_category[r.category]['errors'] += 1
        if r.latency_ms > 0:
            by_category[r.category]['latencies'].append(r.latency_ms)
        if r.score is not None:
            by_category[r.category]['scores'].append(r.score)

    analysis['by_category'] = {}
    for cat, data in by_category.items():
        cat_analysis = {
            'total': data['total'],
            'errors': data['errors']
        }
        if data['latencies']:
            cat_analysis['avg_latency_ms'] = round(sum(data['latencies']) / len(data['latencies']), 1)
        if data['scores']:
            cat_analysis['avg_score'] = round(sum(data['scores']) / len(data['scores']), 2)
        analysis['by_category'][cat] = cat_analysis

    return analysis


def find_model_dirs(base_dir: str) -> List[Tuple[str, str]]:
    """Find all model result directories under a base directory."""
    model_dirs = []

    # Check if base_dir is itself a model directory
    if os.path.exists(os.path.join(base_dir, 'questions.jsonl')):
        name = os.path.basename(base_dir)
        return [(name, base_dir)]

    # Look for model directories
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path):
            questions_file = os.path.join(path, 'questions.jsonl')
            if os.path.exists(questions_file):
                model_dirs.append((entry, path))
            else:
                # Recurse one level
                for sub_entry in os.listdir(path):
                    sub_path = os.path.join(path, sub_entry)
                    if os.path.isdir(sub_path):
                        if os.path.exists(os.path.join(sub_path, 'questions.jsonl')):
                            model_dirs.append((sub_entry, sub_path))

    return sorted(model_dirs)


def compare_runs(base_dir: str) -> Dict:
    """Compare multiple model runs."""
    model_dirs = find_model_dirs(base_dir)

    if not model_dirs:
        return {'error': 'No model results found'}

    comparisons = []

    for name, path in model_dirs:
        analysis = analyze_single_run(path)
        meta = load_run_meta(path)

        entry = {
            'model': name,
            'path': path,
            'analysis': analysis
        }

        if meta:
            entry['engine'] = meta.get('engine', {}).get('name', 'unknown')
            entry['gpu'] = meta.get('gpu', {}).get('name', 'unknown')
            entry['timestamp'] = meta.get('timestamp', '')

        comparisons.append(entry)

    return {
        'models_compared': len(comparisons),
        'results': comparisons
    }


def print_single_analysis(analysis: Dict, detail: bool = False):
    """Print analysis for a single run."""
    if 'error' in analysis:
        print(f"[ERROR] {analysis['error']}")
        return

    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    print(f"\nQuestions: {analysis['successful']}/{analysis['total_questions']} successful")
    if analysis['errors'] > 0:
        print(f"Errors: {analysis['errors']} ({analysis['error_rate']}%)")

    if 'latency_ms' in analysis:
        lat = analysis['latency_ms']
        print(f"\nLatency (ms):")
        print(f"  Min: {lat['min']}, Max: {lat['max']}, Avg: {lat['avg']}")
        if lat.get('p50'):
            print(f"  P50: {lat['p50']}, P95: {lat.get('p95', 'N/A')}")

    if 'tokens_per_sec' in analysis:
        tps = analysis['tokens_per_sec']
        print(f"\nTokens/sec:")
        print(f"  Min: {tps['min']}, Max: {tps['max']}, Avg: {tps['avg']}")

    if 'time_to_first_token_ms' in analysis:
        ttft = analysis['time_to_first_token_ms']
        print(f"\nTime to First Token (ms):")
        print(f"  Min: {ttft['min']}, Max: {ttft['max']}, Avg: {ttft['avg']}")

    if 'scoring' in analysis:
        sc = analysis['scoring']
        print(f"\nScoring:")
        print(f"  Scored: {sc['scored']}/{analysis['total_questions']}")
        print(f"  Total Score: {sc['total_score']}/{sc['max_possible']} ({sc['percentage']}%)")
        print(f"  Average: {sc['avg_score']}/3")

    if detail and 'by_category' in analysis:
        print(f"\nBy Category:")
        print("-" * 50)
        print(f"{'Category':<25} {'Total':>6} {'Errors':>7} {'Avg ms':>8} {'Avg Score':>10}")
        print("-" * 50)

        for cat, data in sorted(analysis['by_category'].items()):
            lat_str = f"{data.get('avg_latency_ms', 'N/A'):>8}" if 'avg_latency_ms' in data else '     N/A'
            score_str = f"{data.get('avg_score', 'N/A'):>10}" if 'avg_score' in data else '       N/A'
            print(f"{cat:<25} {data['total']:>6} {data['errors']:>7} {lat_str} {score_str}")

    print("=" * 60)


def print_comparison(comparison: Dict):
    """Print comparison of multiple runs."""
    if 'error' in comparison:
        print(f"[ERROR] {comparison['error']}")
        return

    print("\n" + "=" * 100)
    print("COMPARISON OF MODEL RESULTS")
    print("=" * 100)
    print(f"Models compared: {comparison['models_compared']}")
    print("-" * 100)

    # Header
    print(f"{'Model':<30} {'Q':<5} {'Err':<4} {'Avg ms':<10} {'Tok/s':<8} {'Score':<10} {'%':>6}")
    print("-" * 100)

    for entry in comparison['results']:
        model = entry['model'][:30]
        analysis = entry['analysis']

        if 'error' in analysis:
            print(f"{model:<30} Error: {analysis['error']}")
            continue

        total = analysis['total_questions']
        errors = analysis['errors']

        avg_lat = 'N/A'
        if 'latency_ms' in analysis:
            avg_lat = f"{analysis['latency_ms']['avg']:.0f}"

        avg_tps = 'N/A'
        if 'tokens_per_sec' in analysis:
            avg_tps = f"{analysis['tokens_per_sec']['avg']:.1f}"

        score_str = 'N/A'
        pct_str = 'N/A'
        if 'scoring' in analysis:
            sc = analysis['scoring']
            score_str = f"{sc['total_score']}/{sc['max_possible']}"
            pct_str = f"{sc['percentage']:.1f}%"

        print(f"{model:<30} {total:<5} {errors:<4} {avg_lat:<10} {avg_tps:<8} {score_str:<10} {pct_str:>6}")

    print("=" * 100)


def export_comparison_csv(comparison: Dict, filepath: str):
    """Export comparison to CSV file."""
    if 'error' in comparison:
        print(f"[ERROR] Cannot export: {comparison['error']}")
        return

    with open(filepath, 'w') as f:
        # Header
        f.write('model,engine,gpu,total_questions,errors,avg_latency_ms,avg_tokens_per_sec,')
        f.write('score,max_score,score_pct,timestamp\n')

        for entry in comparison['results']:
            model = entry['model']
            engine = entry.get('engine', '')
            gpu = entry.get('gpu', '')
            timestamp = entry.get('timestamp', '')
            analysis = entry['analysis']

            if 'error' in analysis:
                f.write(f'{model},{engine},{gpu},error,{analysis["error"]}\n')
                continue

            total = analysis['total_questions']
            errors = analysis['errors']

            avg_lat = analysis.get('latency_ms', {}).get('avg', '')
            avg_tps = analysis.get('tokens_per_sec', {}).get('avg', '')

            score = ''
            max_score = ''
            score_pct = ''
            if 'scoring' in analysis:
                sc = analysis['scoring']
                score = sc['total_score']
                max_score = sc['max_possible']
                score_pct = sc['percentage']

            f.write(f'{model},{engine},{gpu},{total},{errors},{avg_lat},{avg_tps},')
            f.write(f'{score},{max_score},{score_pct},{timestamp}\n')

    print(f"Exported to: {filepath}")


def main():
    """Main entry point."""
    args = parse_args()

    if not os.path.exists(args.results_dir):
        print(f"[ERROR] Directory not found: {args.results_dir}")
        return 1

    if args.compare:
        comparison = compare_runs(args.results_dir)

        if args.json:
            print(json.dumps(comparison, indent=2))
        else:
            print_comparison(comparison)

        if args.export:
            export_comparison_csv(comparison, args.export)

    else:
        # Single run analysis
        if os.path.exists(os.path.join(args.results_dir, 'questions.jsonl')):
            model_dir = args.results_dir
        else:
            # Find first model directory
            model_dirs = find_model_dirs(args.results_dir)
            if model_dirs:
                model_dir = model_dirs[0][1]
            else:
                print("[ERROR] No results found in directory")
                return 1

        analysis = analyze_single_run(model_dir)

        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print_single_analysis(analysis, detail=args.detail or args.by_category)

    return 0


if __name__ == '__main__':
    sys.exit(main())
