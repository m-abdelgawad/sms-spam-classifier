"""
Module for configuring logging for the SMS Spam Classifier.

This module ensures logs are written to both the console and a log file.
The log file is stored in the `logs` directory with a timestamped filename.
"""

import logging
import os
from datetime import datetime


def setup_logger(level: str = "INFO") -> logging.Logger:
    """
    Set up and return a logger with console and file handlers.

    Args:
        level (str): The logging level to use (e.g., "INFO", "DEBUG").

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Initialize the logger with the current module's name
    logger = logging.getLogger("sms-spam-classifier")

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        return logger

    # Set the log level based on the argument
    numeric_level = getattr(logging, level.upper(), None)
    logger.setLevel(numeric_level)

    # Create a logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "../../logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Generate a timestamped filename for the log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"sms-spam-classifier_{timestamp}.log")

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)

    # Create a stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Define a log format for readability
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
