import argparse
import logging
import subprocess
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import config_loader
from src.utils import logging_utils


def get_run_output_dir(base_dir, run_id):
    """Creates and returns the output directory for a specific run."""
    run_output_dir = Path(base_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def construct_args(config, stage_args_map):
    """Constructs list of command line args from config based on map."""
    args_list = []
    for config_key, details in stage_args_map.items():
        if config_key in config:
            value = config[config_key]
            config_arg = details.get("arg")
            is_flag = details.get("is_flag", False)

            if isinstance(value, bool):
                if value and is_flag:
                    args_list.append(config_arg)
            elif isinstance(value, list):
                if config_arg:
                    args_list.append(config_arg)
                    args_list.extend([str(v) for v in value])
            elif value is not None:
                if config_arg:
                    args_list.append(config_arg)
                    args_list.append(str(value))
    return args_list


def run_stage(stage_name, script_path, config, extra_args=None):
    """Runs a specific stage script using subprocess."""
    logging.info(f"\n--- Running Stage: {stage_name.upper()} ---")
    module_path_parts = ["src"] + list(Path(script_path).parts)
    module_path_parts[-1] = module_path_parts[-1].replace(".py", "")
    module_path = ".".join(module_path_parts)

    cmd = [sys.executable, "-m", module_path]
    if extra_args:
        cmd.extend(extra_args)

    logging.info(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, cwd=Path("."))
        logging.info(f"Stage '{stage_name}' stdout:\n{result.stdout}")
        if result.stderr:
            logging.warning(f"Stage '{stage_name}' stderr:\n{result.stderr}")
        logging.info(f"--- Stage {stage_name.upper()} Completed Successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"--- Stage {stage_name.upper()} Failed ---")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        stdout_preview = (
            "\n".join(e.stdout.splitlines()[:10]) if e.stdout else "[No Stdout]"
        )
        stderr_preview = (
            "\n".join(e.stderr.splitlines()[:10]) if e.stderr else "[No Stderr]"
        )
        logging.error(f"Stdout (preview): {stdout_preview}")
        logging.error(f"Stderr (preview): {stderr_preview}")
        return False
    except FileNotFoundError:
        logging.error(
            f"Error: Python executable or script not found. Command: {' '.join(cmd)}"
        )
        return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred running stage {stage_name}: {e}",
            exc_info=True,
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run experimental pipeline stages based on YAML config."
    )
    parser.add_argument(
        "config_file",
        help="Path to the YAML configuration file (e.g., configs/bioasq_llama3_jpv5oxug.yaml).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Specify which stages to run (e.g., generate compute_se). Runs stages marked true in config if omitted.",
    )

    args = parser.parse_args()

    config = config_loader.load_config(args.config_file)
    if config is None:
        sys.exit(1)

    run_id = config["run_id"]
    stages_config = config.get("stages", {})
    base_output_dir = Path(config["base_output_dir"])
    run_output_dir = get_run_output_dir(base_output_dir, run_id)
    log_file = run_output_dir / f"{run_id}_pipeline.log"

    logging_utils.setup_logger(log_file=log_file)

    logging.info(f"Starting experiment pipeline for run: {run_id}")
    logging.info(f"Config loaded from: {args.config_file}")
    logging.info(f"Output directory: {run_output_dir}")

    if args.stages:
        run_flags = {stage: (stage in args.stages) for stage in stages_config}
        logging.info(
            f"Running specific stages requested via command line: {args.stages}"
        )
    else:
        run_flags = stages_config
        logging.info(f"Running stages based on config file flags: {run_flags}")

    gen_args_map = {
        "model_name": {"arg": "--model_name"},
        "base_model": {"arg": "--base_model"},
        "dataset": {"arg": "--dataset"},
        "num_samples": {"arg": "--num_samples"},
        "num_few_shot": {"arg": "--num_few_shot"},
        "temperature": {"arg": "--temperature"},
        "top_p": {"arg": "--top_p"},
        "enable_brief": {"arg": "--enable_brief", "is_flag": True},
        "brief_prompt": {"arg": "--brief_prompt"},
        "use_context": {"arg": "--use_context", "is_flag": True},
        "metric": {"arg": "--metric"},
        "seed": {"arg": "--random_seed"},
        "compute_p_true": {"arg": "--compute_p_true", "is_flag": True},
        "p_true_num_fewshot": {"arg": "--p_true_num_fewshot"},
        "model_max_new_tokens": {"arg": "--model_max_new_tokens"},
        "brief_always": {"arg": "--brief_always", "is_flag": True},
    }
    se_args_map = {
        "entailment_model": {"arg": "--entailment_model"},
        "strict_entailment": {"arg": "--strict_entailment", "is_flag": True},
        "condition_on_question": {"arg": "--condition_on_question", "is_flag": True},
        "num_generations": {"arg": "--use_num_generations"},
        "use_all_generations_se": {"arg": "--use_all_generations", "is_flag": True},
        "num_samples": {"arg": "--num_eval_samples"},
    }
    is_args_map = {
        "probe_test_size": {"arg": "--test_size"},
        "probe_seed": {"arg": "--seed"},
    }
    prep_args_map = {}
    eval_args_map = {
        "analysis_score_columns": {"arg": "--score_columns"},
        "analysis_min_samples_per_subtype": {"arg": "--min_samples"},
        "analysis_save_plots": {"arg": "--save_plots", "is_flag": True},
    }

    success = True
    stages_executed = []

    # Stage 1: Generate Answers
    if run_flags.get("generate", False):
        stages_executed.append("generate")
        gen_args = construct_args(config, gen_args_map)
        gen_args.extend(["--output_dir", str(run_output_dir)])
        if "entity" in config:
            gen_args.extend(["--entity", config["entity"]])
        if "experiment_lot" in config:
            gen_args.extend(["--experiment_lot", config["experiment_lot"]])
        if "debug" in config and config["debug"]:
            gen_args.append("--debug")

        success &= run_stage(
            "generate", "semantic_uncertainty/generate.py", config, gen_args
        )
        if not success:
            sys.exit(1)

    # Stage 2: Compute Semantic Entropy
    if run_flags.get("compute_se", False):
        stages_executed.append("compute_se")
        gen_output_pkl = run_output_dir / "validation_generations.pkl"
        if not gen_output_pkl.exists() and "generate" not in stages_executed:
            logging.warning(
                f"Input file {gen_output_pkl} not found for compute_se stage, and generate stage did not run. Skipping compute_se."
            )
        else:
            se_args = construct_args(config, se_args_map)
            se_args.extend(["--run_dir", str(run_output_dir)])
            if "entity" in config:
                se_args.extend(["--entity", config["entity"]])
            if "experiment_lot" in config:
                se_args.extend(["--experiment_lot", config["experiment_lot"]])
            if "debug" in config and config["debug"]:
                se_args.append("--debug")

            success &= run_stage(
                "compute_se", "semantic_uncertainty/compute_se.py", config, se_args
            )
            if not success:
                sys.exit(1)

    # Stage 3: Probe Internal Signals
    if run_flags.get("probe_is", False):
        stages_executed.append("probe_is")
        script_relative_path = "internal_signals/probe.py"
        gen_output_pkl = run_output_dir / "validation_generations.pkl"
        if not gen_output_pkl.exists() and "generate" not in stages_executed:
            logging.warning(
                f"Input file {gen_output_pkl} not found for probe_is stage, and generate stage did not run. Skipping probe_is."
            )
        else:
            stage_args = construct_args(config, is_args_map)
            stage_args.extend(
                [
                    "--base_dir",
                    str(run_output_dir),
                    "--output_csv",
                    str(run_output_dir / f"{run_id}_internal_signal_results.csv"),
                ]
            )
            positional_args = [run_id]
            full_args = positional_args + stage_args
            success &= run_stage("probe_is", script_relative_path, config, full_args)
            if not success:
                sys.exit(1)

    # Stage 4: Prepare Analysis Dataset
    if run_flags.get("prepare_analysis", False):
        stages_executed.append("prepare_analysis")
        raw_anno_file = (
            Path(config["processed_annotation_dir"])
            / f"{run_id}_annotations_processed.csv"
        )
        is_results_file = run_output_dir / f"{run_id}_internal_signal_results.csv"
        uncertainty_pkl = run_output_dir / "uncertainty_measures.pkl"
        generations_pkl = run_output_dir / "validation_generations.pkl"
        final_output_file = run_output_dir / f"{run_id}_final_analysis_data.csv"

        inputs_ok = True
        if not raw_anno_file.exists():
            logging.error(
                f"Annotation file not found for preparation: {raw_anno_file}. Skipping prepare/evaluate stages."
            )
            inputs_ok = False
        if not uncertainty_pkl.exists():
            logging.error(
                f"Uncertainty measures file not found: {uncertainty_pkl}. Skipping prepare/evaluate stages."
            )
            inputs_ok = False
        if not generations_pkl.exists():
            logging.error(
                f"Generations file not found: {generations_pkl}. Skipping prepare/evaluate stages."
            )
            inputs_ok = False
        if not is_results_file.exists():
            logging.warning(
                f"IS results file not found: {is_results_file}. Final data will lack IS score."
            )
            is_results_path_str = "None"
        else:
            is_results_path_str = str(is_results_file)

        if inputs_ok:
            prep_args = construct_args(config, prep_args_map)
            prep_args.extend(
                [
                    str(raw_anno_file),
                    str(run_output_dir),
                    is_results_path_str,
                    str(final_output_file),
                ]
            )
            success &= run_stage(
                "prepare_analysis",
                "preparation/create_analysis_dataset.py",
                config,
                prep_args,
            )
            if not success:
                sys.exit(1)
        else:
            success = False

    # Stage 5: Evaluate Subtypes
    if run_flags.get("evaluate_subtypes", False):
        if not success and not args.stages:
            logging.warning("Skipping evaluate_subtypes because previous stage failed.")
        else:
            script_relative_path = "analysis/evaluate_subtypes.py"
            analysis_input_file = run_output_dir / f"{run_id}_final_analysis_data.csv"
            plot_output_dir_abs = run_output_dir / config.get(
                "analysis_plot_dir_suffix", "plots"
            )

            if not analysis_input_file.exists():
                logging.error(
                    f"Analysis input file not found: {analysis_input_file}. Skipping evaluate stage."
                )
            else:
                stage_args = construct_args(config, eval_args_map)
                positional_args = [str(analysis_input_file), run_id]
                stage_args.extend(["--plot_dir", str(plot_output_dir_abs)])
                full_args = positional_args + stage_args
                current_stage_success = run_stage(
                    "evaluate_subtypes", script_relative_path, config, full_args
                )
                success &= current_stage_success

    logging.info(f"\n--- Experiment Pipeline for Run {run_id} Finished ---")
    if not success and not args.stages:
        logging.error("One or more critical stages failed.")
        sys.exit(1)
    elif not success and args.stages:
        logging.warning(
            f"One or more explicitly requested stages failed: {args.stages}"
        )
    else:
        logging.info("All requested stages completed.")


if __name__ == "__main__":
    main()
