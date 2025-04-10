import logging
import sys
from pathlib import Path


def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Configures the root logger.

    Args:
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str or Path, optional): Path to a file to save logs. Defaults to None (console only).
    """
    log_format = "%(asctime)s - %(levelname)-8s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = []
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)

    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            handlers.append(file_handler)
            print(f"Logging to file: {log_path}")
        except Exception as e:
            logging.error(
                f"Failed to configure file logging at {log_file}: {e}", exc_info=True
            )

    logging.basicConfig(
        level=log_level, format=log_format, datefmt=date_format, handlers=handlers
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
