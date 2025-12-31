#!/usr/bin/env python3
"""
Manual scoring CLI for the LLM Testing Framework.

Usage:
    python score_results.py results/ollama/           # Score all models
    python score_results.py results/ollama/ --model qwen3-4b
    python score_results.py results/ollama/ summary   # Show scoring summary
    python score_results.py results/ollama/ export    # Export scores to CSV

Interactive scoring allows human evaluation of model responses with
a 0-3 scoring scale.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.results import load_results, ResultsWriter, QuestionResult


SCORE_DESCRIPTIONS = {
    0: "Wrong, hallucinated, or no answer",
    1: "Partially correct or incomplete",
    2: "Mostly correct, minor issues",
    3: "Correct answer with relevant details"
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Score LLM test results manually',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    score    Interactive scoring mode (default)
    summary  Show scoring summary for all models
    export   Export all scores to CSV

Scoring Scale:
    3 = Correct answer with relevant details
    2 = Mostly correct, minor issues
    1 = Partially correct or incomplete
    0 = Wrong, hallucinated, or no answer

Examples:
    # Score all unscored questions interactively
    python score_results.py results/ollama/

    # Score specific model only
    python score_results.py results/ollama/ --model qwen3-4b

    # Re-score already scored questions
    python score_results.py results/ollama/ --rescore

    # View scoring summary
    python score_results.py results/ollama/ summary

    # Export to CSV
    python score_results.py results/ollama/ export --output scores.csv
        """
    )

    parser.add_argument(
        'results_dir',
        help='Results directory to score'
    )
    parser.add_argument(
        'command',
        nargs='?',
        default='score',
        choices=['score', 'summary', 'export'],
        help='Command to run (default: score)'
    )
    parser.add_argument(
        '--model',
        help='Score specific model only'
    )
    parser.add_argument(
        '--category',
        help='Score specific category only'
    )
    parser.add_argument(
        '--rescore',
        action='store_true',
        help='Allow re-scoring of already scored questions'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for export command'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-score using keyword matching (experimental)'
    )

    return parser.parse_args()


def find_model_dirs(base_dir: str) -> List[Tuple[str, str]]:
    """Find all model result directories."""
    model_dirs = []

    if os.path.exists(os.path.join(base_dir, 'questions.jsonl')):
        name = os.path.basename(base_dir)
        return [(name, base_dir)]

    for entry in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path):
            questions_file = os.path.join(path, 'questions.jsonl')
            if os.path.exists(questions_file):
                model_dirs.append((entry, path))

    return model_dirs


def get_unscored_questions(
    results: List[QuestionResult],
    rescore: bool = False,
    category: Optional[str] = None
) -> List[QuestionResult]:
    """Get list of questions that need scoring."""
    unscored = []
    for r in results:
        if category and r.category != category:
            continue
        if rescore or r.score is None:
            unscored.append(r)
    return unscored


def display_question(
    result: QuestionResult,
    index: int,
    total: int
):
    """Display a question for scoring."""
    print("\n" + "=" * 70)
    print(f"Question {index}/{total} [{result.category}]")
    print("=" * 70)

    print(f"\nQ: {result.question}")
    print(f"\nExpected: {result.expected}")

    print("\n" + "-" * 70)
    print("Model Response:")
    print("-" * 70)

    response = result.response or "(No response)"
    # Wrap long lines
    for line in response.split('\n'):
        if len(line) > 68:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > 68:
                    print(current_line)
                    current_line = word
                else:
                    current_line = current_line + " " + word if current_line else word
            if current_line:
                print(current_line)
        else:
            print(line)

    print("-" * 70)

    if result.error:
        print(f"\n[ERROR] {result.error}")

    if result.score is not None:
        print(f"\nCurrent score: {result.score} ({SCORE_DESCRIPTIONS[result.score]})")

    print(f"\nLatency: {result.latency_ms:.0f}ms | Tokens: {result.tokens_generated}")


def get_score_input() -> Tuple[Optional[int], Optional[str]]:
    """Get score input from user."""
    print("\nScoring scale:")
    for score, desc in SCORE_DESCRIPTIONS.items():
        print(f"  {score} = {desc}")
    print("  s = skip | q = quit")

    while True:
        try:
            inp = input("\nScore (0-3, s=skip, q=quit): ").strip().lower()

            if inp == 'q':
                return None, 'quit'
            if inp == 's':
                return None, 'skip'
            if inp in ['0', '1', '2', '3']:
                score = int(inp)
                notes = input("Notes (optional): ").strip()
                return score, notes if notes else None

            print("Invalid input. Enter 0-3, s, or q.")

        except (EOFError, KeyboardInterrupt):
            return None, 'quit'


def score_model(
    model_name: str,
    model_dir: str,
    rescore: bool = False,
    category: Optional[str] = None
) -> Dict:
    """Score a single model's results."""
    questions_file = os.path.join(model_dir, 'questions.jsonl')

    if not os.path.exists(questions_file):
        return {'error': 'No results found'}

    results = load_results(questions_file)
    unscored = get_unscored_questions(results, rescore, category)

    if not unscored:
        return {'scored': 0, 'skipped': 0, 'message': 'All questions already scored'}

    writer = ResultsWriter(os.path.dirname(model_dir), model_name)

    print(f"\n{'=' * 70}")
    print(f"SCORING: {model_name}")
    print(f"{'=' * 70}")
    print(f"Questions to score: {len(unscored)}")

    scored_count = 0
    skipped_count = 0

    for i, result in enumerate(unscored, 1):
        display_question(result, i, len(unscored))

        score, notes = get_score_input()

        if notes == 'quit':
            print("\nExiting scorer...")
            break
        elif notes == 'skip' or score is None:
            skipped_count += 1
            continue
        else:
            success = writer.update_result_score(result.q_id, score, notes)
            if success:
                scored_count += 1
                print(f"Saved score {score} for question {result.q_id}")
            else:
                print(f"[ERROR] Failed to save score")

    return {
        'scored': scored_count,
        'skipped': skipped_count,
        'remaining': len(unscored) - scored_count - skipped_count
    }


def print_summary(base_dir: str):
    """Print scoring summary for all models."""
    model_dirs = find_model_dirs(base_dir)

    if not model_dirs:
        print("No model results found")
        return

    print("\n" + "=" * 80)
    print("SCORING SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'Scored':<10} {'Score':<12} {'Max':<8} {'%':>6}")
    print("-" * 80)

    for name, path in model_dirs:
        results = load_results(os.path.join(path, 'questions.jsonl'))

        if not results:
            print(f"{name:<30} No results")
            continue

        total = len(results)
        scored = [r for r in results if r.score is not None]
        scores = [r.score for r in scored]

        if scores:
            total_score = sum(scores)
            max_score = total * 3
            pct = total_score / max_score * 100
            print(f"{name:<30} {len(scored)}/{total:<6} {total_score:<12} {max_score:<8} {pct:>5.1f}%")
        else:
            print(f"{name:<30} {0}/{total:<6} Not scored")

    print("=" * 80)

    # By category summary
    print("\nBy Category (all models combined):")
    print("-" * 60)
    print(f"{'Category':<25} {'Scored':<8} {'Avg Score':<12} {'Score/Max':<12}")
    print("-" * 60)

    by_cat = {}
    for name, path in model_dirs:
        results = load_results(os.path.join(path, 'questions.jsonl'))
        for r in results:
            if r.category not in by_cat:
                by_cat[r.category] = {'total': 0, 'scores': []}
            by_cat[r.category]['total'] += 1
            if r.score is not None:
                by_cat[r.category]['scores'].append(r.score)

    for cat in sorted(by_cat.keys()):
        data = by_cat[cat]
        if data['scores']:
            avg = sum(data['scores']) / len(data['scores'])
            total_score = sum(data['scores'])
            max_score = len(data['scores']) * 3
            print(f"{cat:<25} {len(data['scores']):<8} {avg:<12.2f} {total_score}/{max_score}")
        else:
            print(f"{cat:<25} 0        Not scored")


def export_scores(base_dir: str, output_file: str):
    """Export all scores to CSV."""
    model_dirs = find_model_dirs(base_dir)

    if not model_dirs:
        print("No model results found")
        return

    with open(output_file, 'w') as f:
        # Header
        f.write('model,q_id,category,question,expected,response,score,scorer_notes,latency_ms\n')

        for name, path in model_dirs:
            results = load_results(os.path.join(path, 'questions.jsonl'))

            for r in results:
                # Escape CSV fields
                question = r.question.replace('"', '""')
                expected = r.expected.replace('"', '""')
                response = (r.response or '').replace('"', '""')
                notes = (r.scorer_notes or '').replace('"', '""')

                score = r.score if r.score is not None else ''

                f.write(f'{name},{r.q_id},{r.category},"{question}","{expected}",')
                f.write(f'"{response}",{score},"{notes}",{r.latency_ms:.0f}\n')

    print(f"Exported scores to: {output_file}")


def auto_score(result: QuestionResult) -> Tuple[int, str]:
    """
    Auto-score using simple keyword matching (experimental).

    This is a basic heuristic - human review is recommended.
    """
    expected_lower = result.expected.lower()
    response_lower = (result.response or '').lower()

    # Check for "not found" type responses
    not_found_phrases = ['not found', 'no ', 'does not', "doesn't", 'cannot', 'unable']
    if any(p in expected_lower for p in not_found_phrases):
        if any(p in response_lower for p in not_found_phrases):
            return 3, "auto: correct negative response"
        return 0, "auto: missed negative case"

    # Extract key values from expected
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?', expected_lower)
    words = re.findall(r'\b[a-z]{3,}\b', expected_lower)

    # Check how many key terms appear in response
    matches = 0
    for num in numbers:
        if num in response_lower:
            matches += 1
    for word in words:
        if word in response_lower:
            matches += 1

    total_keys = len(numbers) + len(words)
    if total_keys == 0:
        return 1, "auto: no keywords to match"

    match_ratio = matches / total_keys

    if match_ratio >= 0.7:
        return 3, f"auto: {match_ratio:.0%} keyword match"
    elif match_ratio >= 0.5:
        return 2, f"auto: {match_ratio:.0%} keyword match"
    elif match_ratio >= 0.2:
        return 1, f"auto: {match_ratio:.0%} keyword match"
    else:
        return 0, f"auto: {match_ratio:.0%} keyword match"


def main():
    """Main entry point."""
    args = parse_args()

    if not os.path.exists(args.results_dir):
        print(f"[ERROR] Directory not found: {args.results_dir}")
        return 1

    if args.command == 'summary':
        print_summary(args.results_dir)
        return 0

    elif args.command == 'export':
        output = args.output or 'scores.csv'
        export_scores(args.results_dir, output)
        return 0

    elif args.command == 'score':
        model_dirs = find_model_dirs(args.results_dir)

        if args.model:
            # Filter to specific model
            model_dirs = [(n, p) for n, p in model_dirs if args.model in n]
            if not model_dirs:
                print(f"[ERROR] Model not found: {args.model}")
                return 1

        if not model_dirs:
            print("No model results found")
            return 1

        print(f"\nFound {len(model_dirs)} model(s) to score")

        for name, path in model_dirs:
            if args.auto:
                # Auto-scoring mode
                results = load_results(os.path.join(path, 'questions.jsonl'))
                writer = ResultsWriter(os.path.dirname(path), name)

                for r in results:
                    if r.score is None or args.rescore:
                        score, notes = auto_score(r)
                        writer.update_result_score(r.q_id, score, notes)

                print(f"Auto-scored {name}")
            else:
                # Interactive scoring
                result = score_model(name, path, args.rescore, args.category)

                if 'error' in result:
                    print(f"[ERROR] {name}: {result['error']}")
                else:
                    print(f"\n{name}: Scored {result['scored']}, skipped {result['skipped']}")

        return 0


if __name__ == '__main__':
    sys.exit(main())
