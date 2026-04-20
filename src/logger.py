import os
import logging

from termcolor import colored


def setup_logging(output_dir: str, logger_name: str = "wukekever") -> logging.Logger:
    """
    Set up console and file logging.

    The console format is intentionally designed to match the desired style:

        [⏳ 2026-04-12 17:15:08][🤖 wukekever]: Step [  300/ 1000] - ...

    Args:
        output_dir: Directory used to save the log file.
        logger_name: Name of the logger instance shown in the log prefix.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "log.txt")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicated handlers if setup_logging is called more than once.
    if logger.handlers:
        logger.handlers.clear()

    file_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_fmt = (
        colored("[⏳ %(asctime)s]", "light_cyan")
        + colored("[🤖 %(name)s]", "blue")
        + colored(": %(message)s", "magenta")
    )  # https://pypi.org/project/termcolor/
    console_formatter = logging.Formatter(
        fmt=console_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
