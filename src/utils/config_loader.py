import yaml
import logging
from pathlib import Path
import os


def load_config(config_path):
    """
    Loads a configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary, or None if the file
        does not exist, is not a valid YAML file, or there is an error loading it.
    """
    path = Path(config_path)
    if not path.is_file():
        logging.error(f"Configuration file not found: {path}")
        return None
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {path}")
        validate_config(config)
        return config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading config {path}: {e}")
        return None


def validate_config(config):
    """
    Validates that the given configuration dictionary has the required keys
    and that the values are of the correct type.

    Required keys:

    - run_id: str
    - stages: dict
    - dataset: str
    - model_name: str
    - base_output_dir: str

    Raises:
    - ValueError: If any of the required keys are missing or if their values
      are of the wrong type.
    """
    required_keys = ["run_id", "stages", "model_name"]

    is_any_qa_stage_active = False
    if isinstance(config.get("stages"), dict):
        is_any_qa_stage_active = any(
            k in ["generate", "compute_se", "probe_is", "train_hybrid_meta", 
                   "calculate_aurocs", "prepare_analysis", "evaluate_subtypes"] 
            and v
            for k,v in config["stages"].items()
        )

    if is_any_qa_stage_active:
        required_keys.append("dataset")

    if not ("base_output_dir" in config or "base_output_dir_synthetic" in config or "base_output_dir_qa" in config) :
         raise ValueError("At least one of 'base_output_dir', 'base_output_dir_synthetic', or 'base_output_dir_qa' must be defined in config.")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in config: {missing_keys}")

    if not isinstance(config.get("stages"), dict):
        raise ValueError("'stages' key must be a dictionary in config.")

    logging.info("Config basic validation passed.")


def expand_env_vars(config):
    """
    Recursively expands environment variables within a given configuration.

    Args:
        config (Union[dict, list, str]): The configuration data which may contain
        environment variables to be expanded. It can be a dictionary, list, 
        or string.

    Returns:
        Union[dict, list, str]: The configuration data with environment 
        variables expanded. The structure of the input is preserved.
    """

    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(i) for i in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config
