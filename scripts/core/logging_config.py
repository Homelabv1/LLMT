"""
Logging configuration for the LLM Testing Framework.

This module provides a consistent logging setup across all modules.
"""

import logging
import sys
from typing import Optional

from .config import DEFAULT_LOG_CONFIG, get_log_level_int


# Module-level logger cache
_loggers = {}


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the framework.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        format_string: Optional custom format string
    """
    config = DEFAULT_LOG_CONFIG

    # Determine log level
    log_level = get_log_level_int(level or config.level)

    # Determine format
    log_format = format_string or config.format
    date_format = config.date_format

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger for 'llmt' namespace
    root_logger = logging.getLogger('llmt')
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file or config.log_to_file:
        file_path = log_file or config.log_file_path
        if file_path:
            try:
                file_handler = logging.FileHandler(file_path)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                root_logger.warning(f"Could not create log file handler: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    Args:
        name: Module name (e.g., 'llmt.runner', 'llmt.engines.ollama')

    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        # Ensure name is under 'llmt' namespace
        if not name.startswith('llmt.'):
            name = f'llmt.{name}'

        logger = logging.getLogger(name)
        _loggers[name] = logger

    return _loggers[name]


class LoggerMixin:
    """
    Mixin class that provides logging functionality to classes.

    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.log.info("Initialized")
    """

    @property
    def log(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


# Convenience functions that use the default logger
_default_logger = None


def _get_default_logger() -> logging.Logger:
    """Get the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger('llmt')
    return _default_logger


def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    _get_default_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    _get_default_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    _get_default_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    _get_default_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    _get_default_logger().critical(msg, *args, **kwargs)
