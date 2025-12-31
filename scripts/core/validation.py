"""
Input validation utilities for the LLM Testing Framework.

This module provides validation functions for user inputs to prevent
injection attacks and ensure data integrity.
"""

import os
import re
from typing import Optional, Tuple

from .config import MODEL_NAME_PATTERN, MODEL_NAME_MAX_LENGTH


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_model_name(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a model name for safety and correctness.

    Args:
        model_name: The model name to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    if not model_name:
        return False, "Model name cannot be empty"

    if len(model_name) > MODEL_NAME_MAX_LENGTH:
        return False, f"Model name exceeds maximum length ({MODEL_NAME_MAX_LENGTH})"

    # Check for valid characters
    if not re.match(MODEL_NAME_PATTERN, model_name):
        return False, (
            "Model name contains invalid characters. "
            "Allowed: alphanumeric, dash, underscore, colon, dot, slash"
        )

    # Check for path traversal attempts
    if '..' in model_name:
        return False, "Model name cannot contain '..'"

    # Check for null bytes
    if '\x00' in model_name:
        return False, "Model name cannot contain null bytes"

    # Check for shell metacharacters
    shell_chars = ['$', '`', '|', ';', '&', '>', '<', '!', '(', ')', '{', '}', '[', ']', '*', '?', '~']
    for char in shell_chars:
        if char in model_name:
            return False, f"Model name cannot contain shell metacharacter '{char}'"

    return True, None


def validate_model_name_strict(model_name: str) -> str:
    """
    Validate a model name and raise an exception if invalid.

    Args:
        model_name: The model name to validate

    Returns:
        The validated model name (unchanged if valid)

    Raises:
        ValidationError: If validation fails
    """
    is_valid, error = validate_model_name(model_name)
    if not is_valid:
        raise ValidationError(f"Invalid model name '{model_name}': {error}")
    return model_name


def validate_file_path(file_path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate a file path for safety.

    Args:
        file_path: The file path to validate
        must_exist: If True, check that file exists

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "File path cannot be empty"

    # Check for null bytes
    if '\x00' in file_path:
        return False, "File path cannot contain null bytes"

    # Normalize and check for path traversal outside expected directories
    try:
        normalized = os.path.normpath(file_path)
        absolute = os.path.abspath(normalized)
    except Exception as e:
        return False, f"Invalid file path: {e}"

    if must_exist and not os.path.exists(absolute):
        return False, f"File does not exist: {file_path}"

    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a URL for safety.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"

    # Basic URL pattern check
    url_pattern = r'^https?://[a-zA-Z0-9][-a-zA-Z0-9._]*[a-zA-Z0-9](:\d+)?(/.*)?$'
    if not re.match(url_pattern, url):
        return False, "Invalid URL format. Must be http:// or https://"

    return True, None


def sanitize_for_path(name: str) -> str:
    """
    Sanitize a name for use in file paths.

    Args:
        name: The name to sanitize

    Returns:
        Sanitized name safe for file paths
    """
    # Replace common separators with safe alternatives
    sanitized = name.replace(':', '-')
    sanitized = sanitized.replace('/', '_')
    sanitized = sanitized.replace('\\', '_')

    # Remove any remaining unsafe characters
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', sanitized)

    return sanitized
