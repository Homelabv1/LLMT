"""
Question loading and validation for the LLM Testing Framework.

This module handles loading test questions from JSON files, validating their
structure, and providing utilities for working with question categories.
"""

import json
import os
from typing import Dict, List, Optional, Union


class Question:
    """Represents a single test question with its metadata."""

    def __init__(
        self,
        q_id: int,
        category: str,
        question: str,
        expected: str,
        difficulty: str
    ):
        """
        Initialize a Question instance.

        Args:
            q_id: Unique question identifier
            category: Question category (e.g., 'simple_lookup', 'aggregation')
            question: The actual question text
            expected: Expected answer content
            difficulty: Difficulty level ('easy', 'medium', 'hard')
        """
        self.q_id = q_id
        self.category = category
        self.question = question
        self.expected = expected
        self.difficulty = difficulty

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'q_id': self.q_id,
            'category': self.category,
            'question': self.question,
            'expected': self.expected,
            'difficulty': self.difficulty
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Question':
        """Create a Question from a dictionary."""
        return cls(
            q_id=data['q_id'],
            category=data['category'],
            question=data['question'],
            expected=data['expected'],
            difficulty=data.get('difficulty', 'medium')
        )

    def __repr__(self) -> str:
        return f"Question(q_id={self.q_id}, category='{self.category}')"


def load_questions(filepath: str) -> List[Question]:
    """
    Load questions from a JSON file.

    Args:
        filepath: Path to the questions JSON file

    Returns:
        List of Question objects

    Raises:
        FileNotFoundError: If the questions file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the JSON structure is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'questions' not in data:
        raise ValueError("Questions file must contain a 'questions' key")

    questions = []
    for q_data in data['questions']:
        questions.append(Question.from_dict(q_data))

    return questions


def validate_questions(questions: List[Question]) -> List[str]:
    """
    Validate a list of questions for completeness and consistency.

    Args:
        questions: List of Question objects to validate

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    seen_ids = set()
    valid_difficulties = {'easy', 'medium', 'hard'}
    valid_categories = {
        'simple_lookup',
        'aggregation',
        'filtering',
        'compatibility',
        'complex_multistep',
        'natural_language',
        'inventory_management',
        'error_handling'
    }

    for q in questions:
        # Check for duplicate IDs
        if q.q_id in seen_ids:
            errors.append(f"Duplicate question ID: {q.q_id}")
        seen_ids.add(q.q_id)

        # Validate required fields
        if not q.question or not q.question.strip():
            errors.append(f"Question {q.q_id}: Empty question text")

        if not q.expected or not q.expected.strip():
            errors.append(f"Question {q.q_id}: Empty expected answer")

        # Validate category
        if q.category not in valid_categories:
            errors.append(
                f"Question {q.q_id}: Invalid category '{q.category}'. "
                f"Valid categories: {valid_categories}"
            )

        # Validate difficulty
        if q.difficulty not in valid_difficulties:
            errors.append(
                f"Question {q.q_id}: Invalid difficulty '{q.difficulty}'. "
                f"Valid difficulties: {valid_difficulties}"
            )

    return errors


def get_questions_by_category(
    questions: List[Question],
    category: Optional[str] = None
) -> Dict[str, List[Question]]:
    """
    Group questions by category.

    Args:
        questions: List of Question objects
        category: Optional specific category to filter by

    Returns:
        Dictionary mapping category names to lists of questions
    """
    result = {}  # type: Dict[str, List[Question]]

    for q in questions:
        if category is not None and q.category != category:
            continue

        if q.category not in result:
            result[q.category] = []
        result[q.category].append(q)

    return result


def get_question_stats(questions: List[Question]) -> Dict:
    """
    Get statistics about a set of questions.

    Args:
        questions: List of Question objects

    Returns:
        Dictionary with question statistics
    """
    by_category = get_questions_by_category(questions)
    by_difficulty = {}  # type: Dict[str, int]

    for q in questions:
        if q.difficulty not in by_difficulty:
            by_difficulty[q.difficulty] = 0
        by_difficulty[q.difficulty] += 1

    return {
        'total': len(questions),
        'by_category': {cat: len(qs) for cat, qs in by_category.items()},
        'by_difficulty': by_difficulty
    }


def load_inventory(filepath: str) -> Dict:
    """
    Load the homelab inventory from a JSON file.

    Args:
        filepath: Path to the inventory JSON file

    Returns:
        Dictionary containing the inventory data

    Raises:
        FileNotFoundError: If the inventory file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Inventory file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_system_prompt(inventory: Dict, template: Optional[str] = None) -> str:
    """
    Create a system prompt with the inventory data embedded.

    Args:
        inventory: The homelab inventory dictionary
        template: Optional custom template (must contain {inventory_json})

    Returns:
        Complete system prompt with inventory embedded
    """
    default_template = """You are a homelab inventory assistant. Answer questions about the following inventory accurately and concisely.

INVENTORY DATA:
{inventory_json}

RULES:
1. Only answer based on the inventory data provided above
2. If information is not in the inventory, say "Not found in inventory"
3. Be concise - give direct answers without unnecessary explanation
4. For counts and aggregations, show your work briefly
5. For compatibility questions, check the compatible_with field in spare_parts"""

    if template is None:
        template = default_template

    inventory_json = json.dumps(inventory, indent=2)
    return template.format(inventory_json=inventory_json)
