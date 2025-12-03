"""
Logging utility module for GMIND SDK.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    module_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for a module with consistent formatting.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses default format.
        module_name: Name of the module. If None, uses __name__ from caller.

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger if not already configured
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(handler)
        root_logger.setLevel(level)

    # Get module-specific logger
    if module_name is None:
        # Try to get caller's module name
        import inspect

        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get("__name__", "GMIND")

    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a module.

    This is a convenience function that ensures logging is set up
    and returns a logger with the appropriate name.

    Args:
        name: Logger name. If None, uses caller's __name__.

    Returns:
        Logger instance
    """
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "GMIND")

    logger = logging.getLogger(name)

    # If root logger not configured, set it up
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()

    return logger
