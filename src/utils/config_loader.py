import yaml
import logging
from pathlib import Path
import os


def load_config(config_path):
    """Loads and validates a YAML config file."""
    path = Path(config_path)
    if not path.is_file():
        logging.error(f"Configuration file not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {path}")
        # Add basic validation if needed (e.g., check for required keys)
        validate_config(config)
        # Expand environment variables if used (optional)
        # config = expand_env_vars(config)
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading config {path}: {e}")
        return None


def validate_config(config):
    """Basic validation of the loaded config dictionary."""
    required_keys = ["run_id", "stages", "dataset", "model_name", "base_output_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in config: {missing_keys}")

    if not isinstance(config.get("stages"), dict):
        raise ValueError("'stages' key must be a dictionary in config.")

    logging.info("Config basic validation passed.")


def expand_env_vars(config):
    """Recursively expands environment variables in config values."""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(i) for i in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config
