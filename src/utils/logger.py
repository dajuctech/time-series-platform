"""
Reusable logger for unified log formatting across the project.
"""

import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Initializes and returns a logger instance.

    Args:
        name (str): Logger name (usually the module name).
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
