import argparse
import logging
import subprocess
import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import config_loader
from src.utils import logging_utils


def get_run_output_dir(base_dir, run_id):
    run_output_dir = Path(base_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def construct_args(config, stage_args_map):
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
    logging.info(f"\n--- Running Stage: {stage_name.upper()} ---")
    script_path_obj = Path(script_path)
    src_dir = script_path_obj.parent
    while src_dir.name != 'src' and src_dir != src_dir.parent:
        src_dir = src_dir.parent
    if src_dir.name != 'src':
        raise FileNotFoundError("Could not find 'src' directory relative to script path.")

    relative_path_parts = script_path_obj.relative_to(src_dir).parts
    module_parts = [part.replace('.py', '') for part in relative_path_parts]
    module_path = ".".join(module_parts)

    full_module_path = f"src.{module_path}"

    cmd = [sys.executable, "-m", full_module_path]
    if extra_args:
        cmd.extend(extra_args)

    logging.info(f"Executing command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=project_root)
        logging.info(f"Stage '{stage_name}' stdout:\n{process.stdout}")
        if process.stderr:
            logging.warning(f"Stage '{stage_name}' stderr:\n{process.stderr}")
        logging.info(f"--- Stage {stage_name.upper()} Completed Successfully ---")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"--- Stage {stage_name.upper()} Failed ---")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: Python executable or script not found. Command: {' '.join(cmd)}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred running stage {stage_name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experimental pipeline stages based on YAML config.")
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    parser.add_argument("--stages", nargs="+", default=None, help="Specify which stages to run.")
    args = parser.parse_args()

    config = config_loader.load_config(args.config_file)
    if config is None: sys.exit(1)

    run_id = config["run_id"]
    stages_config = config.get("stages", {})
    base_output_dir = Path(config["base_output_dir"])
    run_output_dir = get_run_output_dir(base_output_dir, run_id)
    log_file = run_output_dir / f"{run_id}_pipeline.log"

    logging_utils.setup_logger(log_file=log_file)

    logging.info(f"Starting experiment pipeline for run: {run_id}")
    logging.info(f"Config loaded from: {args.config_file}")
    logging.info(f"Output directory: {run_output_dir}")

    run_flags = {stage: (stage in args.stages) for stage in stages_config.keys()} if args.stages else stages_config
    logging.info(f"Running stages based on flags: {run_flags}")

    metric_threshold = config.get("metric_threshold")
    if metric_threshold is None:
        logging.error("Missing required 'metric_threshold' in config YAML.")
        sys.exit(1)

    gen_args_map = { "model_name": {"arg": "--model_name"}, "base_model": {"arg": "--base_model"}, "dataset": {"arg": "--dataset"}, "num_samples": {"arg": "--num_samples"}, "num_few_shot": {"arg": "--num_few_shot"}, "temperature": {"arg": "--temperature"}, "top_p": {"arg": "--top_p"}, "enable_brief": {"arg": "--enable_brief", "is_flag": True}, "brief_prompt": {"arg": "--brief_prompt"}, "use_context": {"arg": "--use_context", "is_flag": True}, "metric": {"arg": "--metric"}, "seed": {"arg": "--random_seed"}, "compute_p_true": {"arg": "--compute_p_true", "is_flag": True}, "p_true_num_fewshot": {"arg": "--p_true_num_fewshot"}, "model_max_new_tokens": {"arg": "--model_max_new_tokens"}, "brief_always": {"arg": "--brief_always", "is_flag": True}, "probe_layers_to_extract": {"arg": "--probe_layers_to_extract"} }
    se_args_map = { "entailment_model": {"arg": "--entailment_model"}, "strict_entailment": {"arg": "--strict_entailment", "is_flag": True}, "condition_on_question": {"arg": "--condition_on_question", "is_flag": True}, "num_generations": {"arg": "--use_num_generations"}, "use_all_generations_se": {"arg": "--use_all_generations", "is_flag": True}, "num_samples": {"arg": "--num_eval_samples"}, "metric": {"arg": "--metric"}, "recompute_accuracy": {"arg":"--recompute_accuracy", "is_flag": True}, "compute_p_ik": {"arg":"--compute_p_ik", "is_flag": True}, "compute_p_ik_answerable": {"arg":"--compute_p_ik_answerable", "is_flag": True}, "compute_p_true": {"arg": "--compute_p_true", "is_flag": True}, "compute_predictive_entropy": {"arg": "--compute_predictive_entropy", "is_flag": True}}
    is_args_map = {
        "probe_classifier": {"arg": "--classifier"},
        "probe_n_splits": {"arg": "--n_splits"},
        "probe_seed": {"arg": "--seed"},
    }

    hybrid_meta_train_args_map = {
        "meta_classifier": {"arg": "--meta_classifier"},
        "meta_n_splits": {"arg": "--n_splits"},
        "meta_seed": {"arg": "--seed"}
    }
    calc_auroc_args_map = {
        "analysis_save_aurac_plots": {"arg": "--save_plots", "is_flag": True},
        "analysis_ece_bins": {"arg": "--ece_bins"}
    }
    prep_args_map = {}
    eval_args_map = { "analysis_score_columns": {"arg": "--score_columns"}, "analysis_min_samples_per_subtype": {"arg": "--min_samples"}, "analysis_save_plots": {"arg": "--save_plots", "is_flag": True} }

    success = True
    stages_executed = []

    if run_flags.get("generate", False):
        stages_executed.append("generate"); gen_args = construct_args(config, gen_args_map); gen_args.extend(["--output_dir", str(run_output_dir)]); gen_args.extend(["--metric_threshold", str(metric_threshold)])
        success &= run_stage("generate", "src/semantic_uncertainty/generate.py", config, gen_args)
        if not success: sys.exit(1)

    if run_flags.get("compute_se", False):
        stages_executed.append("compute_se"); gen_output_pkl = run_output_dir / "validation_generations.pkl"
        if gen_output_pkl.exists():
            se_args = construct_args(config, se_args_map); se_args.extend(["--run_dir", str(run_output_dir)]); se_args.extend(["--metric_threshold", str(metric_threshold)])
            success &= run_stage("compute_se", "src/semantic_uncertainty/compute_se.py", config, se_args)
            if not success: sys.exit(1)
        else: logging.warning("Generate output missing. Skipping compute_se.")

    if run_flags.get("probe_is", False):
        stages_executed.append("probe_is"); script_path = "src/internal_signals/probe.py"
        gen_output_pkl = run_output_dir / "validation_generations.pkl"
        if gen_output_pkl.exists():
             stage_args = construct_args(config, is_args_map)
             positional_args = [run_id]; stage_args.extend(["--base_dir", str(run_output_dir)])
             stage_args.extend(["--metric_threshold", str(metric_threshold)])
             if "--classifier" not in stage_args:
                 default_classifier = config.get("probe_classifier", "logistic")
                 stage_args.extend(["--classifier", default_classifier])

             full_args = positional_args + stage_args
             success &= run_stage("probe_is", script_path, config, full_args)
             if not success: sys.exit(1)
        else: logging.warning(f"Input {gen_output_pkl} missing. Skipping probe_is.")

    if run_flags.get("train_hybrid_meta", False):
        stages_executed.append("train_hybrid_meta"); script_path = "src/hybrid/train_hybrid_meta.py"
        uncertainty_pkl = run_output_dir / "uncertainty_measures.pkl"
        is_scores_all_csv = run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
        if uncertainty_pkl.exists() and is_scores_all_csv.exists():
             stage_args = construct_args(config, hybrid_meta_train_args_map)
             positional_args = [run_id, str(run_output_dir)]
             if "--meta_classifier" not in stage_args:
                 default_meta_classifier = config.get("meta_classifier", "logistic")
                 stage_args.extend(["--meta_classifier", default_meta_classifier])

             full_args = positional_args + stage_args
             success &= run_stage("train_hybrid_meta", script_path, config, full_args)
             if not success: sys.exit(1)
        else: logging.warning("Inputs missing for train_hybrid_meta. Skipping.")

    if run_flags.get("calculate_aurocs", False):
         stages_executed.append("calculate_aurocs"); script_path = "src/analysis/calculate_full_set_aurocs.py"
         uncertainty_pkl = run_output_dir / "uncertainty_measures.pkl"
         is_scores_all_csv = run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
         hybrid_meta_scores_all_csv = run_output_dir / f"{run_id}_hybrid_meta_scores_all.csv"
         if uncertainty_pkl.exists() and is_scores_all_csv.exists() and hybrid_meta_scores_all_csv.exists():
             stage_args = construct_args(config, calc_auroc_args_map)
             positional_args = [run_id, str(run_output_dir)]
             auroc_json_output = run_output_dir / config.get("metrics_output_json", f"{run_id}_full_set_metrics.json")
             stage_args.extend(["--output_json", str(auroc_json_output)])
             if "--seed" not in stage_args:
                  stage_args.extend(["--seed", str(config.get("probe_seed", 42))])
             if "--n_splits" not in stage_args:
                  stage_args.extend(["--n_splits", str(config.get("probe_n_splits", 5))])

             full_args = positional_args + stage_args
             success &= run_stage("calculate_aurocs", script_path, config, full_args)
             if not success: sys.exit(1)
         else:
             logging.warning("Inputs missing for calculate_aurocs. Skipping.")

    if run_flags.get("prepare_analysis", False):
        stages_executed.append("prepare_analysis")
        anno_dir = Path(config.get("processed_annotation_dir", "annotation_data"))
        reviewed_anno_file_name = config.get("annotation_csv_filename", f"{run_id}_annotations_processed.csv")
        reviewed_anno_file = anno_dir / run_id / reviewed_anno_file_name if anno_dir.is_absolute() else project_root / anno_dir / run_id / reviewed_anno_file_name
        is_scores_all_csv = run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
        hybrid_meta_scores_all_csv = run_output_dir / f"{run_id}_hybrid_meta_scores_all.csv"
        uncertainty_pkl = run_output_dir / "uncertainty_measures.pkl"
        generations_pkl = run_output_dir / "validation_generations.pkl"
        final_output_file = run_output_dir / f"{run_id}_final_analysis_data.csv"

        inputs_ok = True
        required_files = {"Reviewed Annotations": reviewed_anno_file, "IS Scores (All)": is_scores_all_csv,
                          "Hybrid Meta Scores (All)": hybrid_meta_scores_all_csv, "Uncertainty PKL": uncertainty_pkl,
                          "Generations PKL": generations_pkl }
        for name, path in required_files.items():
            if not path.exists(): logging.error(f"{name} not found: {path}."); inputs_ok = False

        if inputs_ok:
            positional_args = [str(reviewed_anno_file), str(run_output_dir), str(final_output_file)]
            keyword_args = [
                "--is_scores_all_csv", str(is_scores_all_csv),
                "--hybrid_meta_scores_all_csv", str(hybrid_meta_scores_all_csv)
            ]
            keyword_args_from_config = construct_args(config, prep_args_map)
            full_args = positional_args + keyword_args + keyword_args_from_config

            success &= run_stage("prepare_analysis", "src/preparation/create_analysis_dataset.py", config, full_args)
            if not success: sys.exit(1)
        else:
            logging.error("Skipping prepare_analysis due to missing inputs.")
            success = False

    if run_flags.get("evaluate_subtypes", False):
        stages_executed.append("evaluate_subtypes")
        script_path = "src/analysis/evaluate_subtypes.py"
        analysis_input_file = run_output_dir / f"{run_id}_final_analysis_data.csv"
        plot_dir_suffix = config.get("analysis_plot_dir_suffix", "subtype_plots")
        plot_output_dir_abs = run_output_dir / plot_dir_suffix

        if analysis_input_file.exists():
             stage_args = construct_args(config, eval_args_map)
             positional_args = [str(analysis_input_file), run_id]
             stage_args.extend(["--plot_dir", str(plot_output_dir_abs)])
             full_args = positional_args + stage_args
             success &= run_stage("evaluate_subtypes", script_path, config, full_args)
             if not success: sys.exit(1)
        else:
            logging.warning(f"Input file {analysis_input_file} missing. Skipping evaluate_subtypes.")

    logging.info(f"\n--- Experiment Pipeline for Run {run_id} Finished ---")
    if not success: logging.error("One or more critical stages failed during the run.")
    else: logging.info("All requested stages completed.")
    critical_stages = ['generate', 'compute_se', 'probe_is', 'train_hybrid_meta', 'prepare_analysis']
    if not success and any(run_flags.get(stage) for stage in critical_stages):
        sys.exit(1)


if __name__ == "__main__":
    main()