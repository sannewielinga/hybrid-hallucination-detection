import subprocess
import sys
import os
from pathlib import Path
import logging
import time
import yaml
import argparse

log_format = "%(asctime)s - %(levelname)-8s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
runner_log_file = Path("comparative_analysis_runner.log")
file_handler = logging.FileHandler(runner_log_file, mode="a")
file_handler.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(file_handler)

def modify_config_for_test(original_config_path: Path, test_config_path: Path, num_samples_test: int):
    try:
        with open(original_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        original_num_samples = config_data.get('num_samples')
        config_data['num_samples'] = num_samples_test
        
        if 'num_eval_samples' in config_data:
            config_data['num_eval_samples'] = num_samples_test
            
        with open(test_config_path, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)
        logging.info(f"Created temporary test config: {test_config_path} with num_samples = {num_samples_test}")
        return original_num_samples
    except Exception as e:
        logging.error(f"Error modifying config {original_config_path} for test: {e}", exc_info=True)
        return None

def run_experiment_config(config_path_str: str, is_test_run: bool = False, original_num_samples = None):
    config_path = Path(config_path_str)
    if not config_path.is_file():
        logging.error(f"Config file not found: {config_path}")
        return False

    logging.info(f"\n{'='*80}\nStarting experiment for: {config_path.name}{' (TEST RUN)' if is_test_run else ''}\n{'='*80}")
    
    master_script_path = Path("src/run_experiment.py").resolve()
    cmd = [sys.executable, str(master_script_path), str(config_path)]

    logging.info(f"Executing: {' '.join(cmd)}")
    start_time = time.time()
    process_result = None
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parent)
        process_result = process

        if process.returncode == 0:
            logging.info(f"Successfully completed experiment: {config_path.name}")
            logging.info(f"Stdout for {config_path.name} (last 2000 chars):\n{process.stdout[-2000:]}") 
            if process.stderr:
                 logging.warning(f"Stderr for {config_path.name} (Return Code 0) (last 2000 chars):\n{process.stderr[-2000:]}")
            return True
        else:
            logging.error(f"Experiment FAILED: {config_path.name}")
            logging.error(f"Return Code: {process.returncode}")
            logging.error(f"Stdout for {config_path.name}:\n{process.stdout}")
            logging.error(f"Stderr for {config_path.name}:\n{process.stderr}")
            return False
    except FileNotFoundError:
        logging.error(f"Error: Python executable or master script '{master_script_path}' not found.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred running {config_path.name}: {e}", exc_info=True)
        if process_result:
            logging.error(f"(Partial) Stdout for {config_path.name}:\n{process_result.stdout}")
            logging.error(f"(Partial) Stderr for {config_path.name}:\n{process_result.stderr}")
        return False
    finally:
        end_time = time.time()
        logging.info(f"Experiment {config_path.name} took {end_time - start_time:.2f} seconds.")
        if is_test_run and config_path.exists():
            try:
                os.remove(config_path)
                logging.info(f"Removed temporary test config: {config_path}")
            except Exception as e_clean:
                logging.error(f"Error removing temporary test config {config_path}: {e_clean}")
        logging.info(f"\n{'='*80}\nFinished attempt for: {config_path.name}\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all comparative analysis experiments.")
    parser.add_argument(
        "--test_run_samples", 
        type=int, 
        default=None,
        help="If specified, runs all experiments with this number of samples for testing. Otherwise, uses num_samples from YAML."
    )
    args = parser.parse_args()

    is_test_run_flag = args.test_run_samples is not None
    test_samples_count = args.test_run_samples if is_test_run_flag else 0

    base_configs_directory = Path("configs/analyses/") 
    sub_dirs_to_process = ["bertscore", "instruct"]
    
    yaml_files_to_run_map = {}

    for sub_dir_name in sub_dirs_to_process:
        current_configs_dir = base_configs_directory / sub_dir_name
        if not current_configs_dir.is_dir():
            logging.warning(f"Configs subdirectory not found: {current_configs_dir}. Skipping.")
            continue
        
        yaml_files_in_subdir = sorted(list(current_configs_dir.glob("*.yaml")))
        if yaml_files_in_subdir:
            yaml_files_to_run_map[sub_dir_name] = yaml_files_in_subdir
        else:
            logging.warning(f"No YAML files found in {current_configs_dir} to run.")

    if not yaml_files_to_run_map:
        logging.error(f"No YAML files found in any of the specified subdirectories under {base_configs_directory}. Exiting.")
        sys.exit(1)

    total_configs_to_run = sum(len(files) for files in yaml_files_to_run_map.values())
    logging.info(f"Found a total of {total_configs_to_run} YAML configurations to process.")
    if is_test_run_flag:
        logging.info(f"!!! THIS IS A TEST RUN with num_samples = {test_samples_count} for all experiments !!!")


    overall_success = True
    failed_configs_list = []
    configs_processed_count = 0

    for subdir_name, yaml_files in yaml_files_to_run_map.items():
        logging.info(f"\n--- Processing Subdirectory: {subdir_name} ---")
        for config_file_path_original in yaml_files:
            configs_processed_count += 1
            logging.info(f"\nProcessing config {configs_processed_count}/{total_configs_to_run}: {config_file_path_original.name} from {subdir_name}")
            
            current_config_to_run = config_file_path_original
            temp_test_config_path = None
            original_num_samples_val = None

            if is_test_run_flag:
                temp_test_config_name = f"TEMP_TEST_{config_file_path_original.name}"
                temp_test_config_path = config_file_path_original.parent / temp_test_config_name 
                original_num_samples_val = modify_config_for_test(config_file_path_original, temp_test_config_path, test_samples_count)
                if original_num_samples_val is None:
                    logging.error(f"Could not create test config for {config_file_path_original.name}. Skipping.")
                    failed_configs_list.append(config_file_path_original.name)
                    overall_success = False
                    continue
                current_config_to_run = temp_test_config_path
            
            success_current_run = run_experiment_config(str(current_config_to_run), is_test_run=is_test_run_flag, original_num_samples=original_num_samples_val)
            
            if not success_current_run:
                overall_success = False
                failed_configs_list.append(config_file_path_original.name)
                logging.warning(f"Continuing to next config despite failure in {config_file_path_original.name}")
            

    logging.info("\n--- All Comparative Experiments Attempted ---")
    if overall_success:
        logging.info("All configured experiments completed successfully!")
    else:
        logging.error("One or more experiments FAILED.")
        logging.error("Failed configurations:")
        for failed_config_name in failed_configs_list:
            logging.error(f"  - {failed_config_name}")