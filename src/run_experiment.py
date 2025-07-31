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
    """
    Create the output directory for a specific run.

    The output directory is a subdirectory of the given base directory, with
    the name of the run as its name.

    Args:
        base_dir (str): The base directory for the output directory.
        run_id (str): The name of the run.

    Returns:
        Path: The path to the output directory.
    """
    run_output_dir = Path(base_dir) / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def construct_args(config, stage_args_map):
    """
    Construct the command line arguments for a specific stage of the pipeline.

    Takes a dictionary of configuration options and a dictionary of argument
    specifications, and returns a list of command line arguments to be passed to
    the script for the stage.

    The argument specifications dictionary should have the following structure:

    {
        config_key: {
            "arg": str (name of the command line argument)
            "script_arg_name": str (name of the argument as it appears in the
                                  script that will be called)
            "is_flag": bool (True if the argument is a flag, False otherwise)
            "config_key_override": str (optional, the key to use from the
                                        configuration dictionary instead of the
                                        one specified by the config_key)
        }
    }

    If the configuration key is present in the configuration dictionary, the
    corresponding value will be used to construct the command line argument.
    If the value is a boolean, it will be used as a flag. If the value is a list,
    it will be used to construct multiple command line arguments. If the value
    is a string or other type, it will be used as the value for a single command
    line argument.

    Args:
        config (dict): The configuration dictionary.
        stage_args_map (dict): The dictionary of argument specifications.

    Returns:
        list: The list of command line arguments to be passed to the script.
    """
    args_list = []
    for config_key, details in stage_args_map.items():
        actual_config_key_to_check = details.get("config_key_override", config_key)

        if actual_config_key_to_check in config:
            value = config[actual_config_key_to_check]
            script_arg = details.get("script_arg_name", details.get("arg"))
            is_flag = details.get("is_flag", False)

            if script_arg is None: 
                continue

            if isinstance(value, bool):
                if value and is_flag:
                    args_list.append(script_arg)
            elif isinstance(value, list):
                args_list.append(script_arg)
                args_list.extend([str(v) for v in value])
            elif value is not None:
                args_list.append(script_arg)
                args_list.append(str(value))
    return args_list


def run_stage(stage_name, script_module_path, config_unused, extra_args=None):
    """
    Run a stage of the pipeline.

    Args:
        stage_name (str): The name of the stage.
        script_module_path (str): The path to the script module to run.
        config_unused (dict): The configuration dictionary. Unused.
        extra_args (list): Extra command line arguments to pass to the script.
                           Defaults to None.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    logging.info(f"\n--- Running Stage: {stage_name.upper()} ---")
    
    cmd = [sys.executable, "-m", script_module_path]
    if extra_args:
        cmd.extend(extra_args)

    logging.info(f"Executing command: {' '.join(cmd)}")
    process_env = os.environ.copy()
    project_root_str = str(project_root)
    current_pythonpath = process_env.get("PYTHONPATH", "")
    if project_root_str not in current_pythonpath.split(os.pathsep):
        if current_pythonpath:
            process_env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{current_pythonpath}"
        else:
            process_env["PYTHONPATH"] = project_root_str

    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, 
                                 cwd=project_root, env=process_env)
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
        logging.error(f"Error: Python executable or script module not found. Command: {' '.join(cmd)}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred running stage {stage_name}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run experimental pipeline stages based on YAML config.")
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    parser.add_argument("--stages", nargs="+", default=None, help="Specify which stages to run.")
    cli_args = parser.parse_args()

    config = config_loader.load_config(cli_args.config_file)
    if config is None: sys.exit(1)

    run_id = config["run_id"]
    stages_config = config.get("stages", {})
    
    is_any_synthetic_stage_active = any(
        k.startswith("generate_synthetic") or k.endswith("_on_synthetic") 
        for k, v in stages_config.items() if v and (cli_args.stages is None or k in cli_args.stages)
    )
    
    is_any_qa_stage_active = any(
        k in ["generate", "compute_se", "probe_is", "train_hybrid_meta", "calculate_aurocs", "prepare_analysis", "evaluate_subtypes"]
        for k, v in stages_config.items() if v and (cli_args.stages is None or k in cli_args.stages)
    )

    if is_any_synthetic_stage_active and "base_output_dir_synthetic" in config:
         base_output_dir_for_run = Path(config["base_output_dir_synthetic"])
    elif is_any_qa_stage_active and "base_output_dir" in config:
         base_output_dir_for_run = Path(config["base_output_dir"])
    elif "base_output_dir" in config:
        base_output_dir_for_run = Path(config["base_output_dir"])
    else:
        logging.error("A 'base_output_dir' or 'base_output_dir_synthetic' must be defined in the config.")
        sys.exit(1)

    current_run_main_output_dir = get_run_output_dir(base_output_dir_for_run, run_id)
    log_file = current_run_main_output_dir / f"{run_id}_pipeline.log" 
    logging_utils.setup_logger(log_file=log_file)

    logging.info(f"Starting experiment pipeline for run: {run_id}")
    logging.info(f"Config loaded from: {cli_args.config_file}")
    logging.info(f"Main output directory for this run: {current_run_main_output_dir}")

    run_flags = {stage: (stage in cli_args.stages) for stage in stages_config.keys()} if cli_args.stages else stages_config
    logging.info(f"Running stages based on flags: {run_flags}")

    gen_args_map = { "model_name": {"arg": "--model_name"}, "base_model": {"arg": "--base_model"}, "dataset": {"arg": "--dataset"}, "num_samples": {"arg": "--num_samples"}, "num_few_shot": {"arg": "--num_few_shot"}, "temperature": {"arg": "--temperature"}, "top_p": {"arg": "--top_p"}, "enable_brief": {"arg": "--enable_brief", "is_flag": True}, "brief_prompt": {"arg": "--brief_prompt"}, "use_context": {"arg": "--use_context", "is_flag": True}, "metric": {"arg": "--metric"}, "seed": {"arg": "--random_seed"}, "compute_p_true": {"arg": "--compute_p_true", "is_flag": True}, "p_true_num_fewshot": {"arg": "--p_true_num_fewshot"}, "model_max_new_tokens": {"arg": "--model_max_new_tokens"}, "brief_always": {"arg": "--brief_always", "is_flag": True} , "probe_layers_to_extract": {"arg": "--probe_layers_to_extract"}}
    se_args_map = { "entailment_model": {"arg": "--entailment_model"}, "strict_entailment": {"arg": "--strict_entailment", "is_flag": True}, "condition_on_question": {"arg": "--condition_on_question", "is_flag": True}, "num_generations": {"config_key_override": "num_generations", "script_arg_name": "--use_num_generations"}, "use_all_generations_se": {"arg": "--use_all_generations_se", "is_flag": True}, "num_samples": {"config_key_override": "num_samples","script_arg_name": "--num_eval_samples"}, "probe_accuracy_threshold": {"arg": "--probe_accuracy_threshold"}, "compute_p_ik": {"arg": "--compute_p_ik", "is_flag":True}, "compute_p_ik_answerable": {"arg": "--compute_p_ik_answerable", "is_flag":True} }
    is_args_map = { "probe_classifier": {"script_arg_name": "--classifier"}, "probe_n_splits": {"script_arg_name": "--n_splits"}, "probe_seed": {"config_key_override": "probe_seed", "script_arg_name": "--seed"}, "probe_accuracy_threshold": {"script_arg_name": "--probe_accuracy_threshold"} } # Use global seed for probe_seed
    hybrid_meta_train_args_map = { "meta_classifier": {"script_arg_name": "--meta_classifier"}, "probe_n_splits": {"script_arg_name": "--n_splits"}, "seed": {"script_arg_name": "--seed"}} 
    calc_auroc_args_map = { "analysis_save_aurac_plots": {"script_arg_name": "--save_plots", "is_flag": True}, "ece_bins": {"script_arg_name": "--ece_bins"}, "seed": {"script_arg_name":"--seed"}}
    prep_args_map = {}
    eval_args_map = { "analysis_score_columns": {"script_arg_name": "--score_columns"}, "analysis_min_samples_per_subtype": {"script_arg_name": "--min_samples"}, "analysis_save_plots": {"script_arg_name": "--save_plots", "is_flag": True} }

    synth_prompt_gen_args_map = { "synthetic_num_per_med": {"script_arg_name": "--num_per_med"} }
    synth_answer_gen_args_map = { "model_name": {"script_arg_name": "--model_name"}, "base_model": {"script_arg_name": "--base_model"}, "model_max_new_tokens_low_t_synthetic": {"script_arg_name": "--model_max_new_tokens_low_t"}, "model_max_new_tokens": {"script_arg_name": "--model_max_new_tokens"}, "seed": {"script_arg_name": "--random_seed"}, "num_generations_synthetic": {"config_key_override": "num_generations_synthetic", "script_arg_name":"--num_generations"}, "temperature_synthetic": {"script_arg_name": "--temperature_synthetic"}, "top_p": {"script_arg_name": "--top_p"}, "num_samples_synthetic": {"script_arg_name": "--num_samples_synthetic"} }

    success = True
    
    synthetic_prompts_cache_dir = current_run_main_output_dir / "synthetic_prompts_cache"
    num_per_med_val_for_fname = config.get("synthetic_num_per_med", 50)
    prompts_fname_template_val = config.get("synthetic_prompts_filename_template", "prompts_n{num_per_med}.jsonl")
    current_synthetic_prompts_file = synthetic_prompts_cache_dir / prompts_fname_template_val.format(num_per_med=num_per_med_val_for_fname)
    
    synthetic_model_name_slug = config.get("model_name", "unknown_model").replace('/', '_')
    synthetic_model_specific_output_dir = current_run_main_output_dir / "synthetic_model_outputs" / synthetic_model_name_slug

    synthetic_gen_pkl_fname = f"synth_{synthetic_model_name_slug}_{current_synthetic_prompts_file.stem}_generations_synthetic_dosage.pkl"
    synthetic_uncertainty_fname = "uncertainty_measures_synthetic_dosage.pkl"
    synthetic_is_scores_fname = f"{run_id}_internal_signal_scores_synthetic_dosage.csv"
    synthetic_hybrid_meta_fname = f"{run_id}_hybrid_meta_scores_synthetic_dosage.csv"
    synthetic_auroc_json_fname_template = config.get("synthetic_metrics_output_json_template", "{run_id}_synthetic_full_set_metrics.json")
    synthetic_auroc_json_fname = synthetic_auroc_json_fname_template.format(run_id=run_id)

    if run_flags.get("generate_synthetic_dosage_prompts"):
        synthetic_prompts_cache_dir.mkdir(parents=True, exist_ok=True)
        stage_args = construct_args(config, synth_prompt_gen_args_map)
        stage_args.extend(["--output_file", str(current_synthetic_prompts_file)])
        success &= run_stage("generate_synthetic_dosage_prompts", "src.synthetic.generate_dosage_prompts", config, stage_args)
        if not success:
            sys.exit(1)

    if run_flags.get("generate_synthetic_dosage_answers"):
        if not current_synthetic_prompts_file.exists():
            logging.error(f"Synthetic prompts file {current_synthetic_prompts_file} not found.")
            sys.exit(1)
        synthetic_model_specific_output_dir.mkdir(parents=True, exist_ok=True)
        stage_args = construct_args(config, synth_answer_gen_args_map)
        stage_args.extend(["--input_synthetic_prompts", str(current_synthetic_prompts_file)])
        stage_args.extend(["--output_dir_synthetic", str(synthetic_model_specific_output_dir)])
        success &= run_stage("generate_synthetic_dosage_answers", "src.synthetic.generate_synthetic_answers", config, stage_args)
        if not success:
            sys.exit(1)

    if run_flags.get("compute_se_on_synthetic"):
        if not (synthetic_model_specific_output_dir / synthetic_gen_pkl_fname).exists():
            logging.error(f"Synthetic generations PKL {synthetic_model_specific_output_dir / synthetic_gen_pkl_fname} not found for SE.")
            sys.exit(1)
        stage_args = construct_args(config, se_args_map) 
        stage_args.extend([
            "--run_dir", str(synthetic_model_specific_output_dir),
            "--input_generations_filename_override", synthetic_gen_pkl_fname,
            "--output_uncertainty_filename_override", synthetic_uncertainty_fname
        ])
        success &= run_stage("compute_se_on_synthetic", "src.semantic_uncertainty.compute_se", config, stage_args)
        if not success:
            sys.exit(1)

    if run_flags.get("probe_is_on_synthetic"):
        probe_input_gen_file = synthetic_model_specific_output_dir / synthetic_gen_pkl_fname
        if not probe_input_gen_file.exists():
            logging.error(f"Synthetic generations PKL {probe_input_gen_file} missing for IS probe.")
            sys.exit(1)
        stage_args = construct_args(config, is_args_map)
        positional_args_is = [f"{run_id}_synth_is"] 
        stage_args.extend([
            "--base_dir", str(synthetic_model_specific_output_dir),
            "--input_generations_filename_override", synthetic_gen_pkl_fname,
            "--output_scores_filename_override", synthetic_is_scores_fname,
            "--output_model_filename_override", f"{run_id}_probe_model_synthetic_dosage.pkl"
        ])
        success &= run_stage("probe_is_on_synthetic", "src.internal_signals.probe", config, positional_args_is + stage_args)
        if not success:
            sys.exit(1)

    if run_flags.get("train_hybrid_meta_on_synthetic"):
        unc_file = synthetic_model_specific_output_dir / synthetic_uncertainty_fname
        is_file = synthetic_model_specific_output_dir / synthetic_is_scores_fname
        gen_file_for_hybrid_meta = synthetic_model_specific_output_dir / synthetic_gen_pkl_fname 

        if not unc_file.exists() or not is_file.exists() or not gen_file_for_hybrid_meta.exists():
            logging.error(f"Input files for synthetic hybrid meta missing in {synthetic_model_specific_output_dir}. Check paths for uncertainty, IS scores, and generations.")
            if not unc_file.exists():
                logging.error(f"Missing: {unc_file}")
            if not is_file.exists():
                logging.error(f"Missing: {is_file}")
            if not gen_file_for_hybrid_meta.exists():
                logging.error(f"Missing: {gen_file_for_hybrid_meta}")
            sys.exit(1)
            
        stage_args = construct_args(config, hybrid_meta_train_args_map)
        positional_args_hm = [f"{run_id}_synth_hm", str(synthetic_model_specific_output_dir)]
        stage_args.extend([
            "--input_uncertainty_filename_override", synthetic_uncertainty_fname,
            "--input_is_scores_filename_override", synthetic_is_scores_fname,
            "--input_generations_filename_override", synthetic_gen_pkl_fname,
            "--output_hybrid_meta_filename_override", synthetic_hybrid_meta_fname,
            "--output_model_filename_override", f"{run_id}_hybrid_meta_model_synthetic_dosage.pkl", 
            "--output_scalers_filename_override", f"{run_id}_hybrid_meta_scalers_synthetic_dosage.pkl"
        ])
        success &= run_stage("train_hybrid_meta_on_synthetic", "src.hybrid.train_hybrid_meta", config, positional_args_hm + stage_args)
        if not success:
            sys.exit(1)
        
    if run_flags.get("calculate_aurocs_on_synthetic"):
        required_files_exist = True
        if not (synthetic_model_specific_output_dir / synthetic_uncertainty_fname).exists():
            logging.warning(f"Synthetic uncertainty file missing: {synthetic_uncertainty_fname}"); required_files_exist=False
        if run_flags.get("probe_is_on_synthetic", False) and not (synthetic_model_specific_output_dir / synthetic_is_scores_fname).exists():
            logging.warning(f"Synthetic IS scores file missing: {synthetic_is_scores_fname}"); required_files_exist=False
        if run_flags.get("train_hybrid_meta_on_synthetic", False) and not (synthetic_model_specific_output_dir / synthetic_hybrid_meta_fname).exists():
            logging.warning(f"Synthetic Hybrid Meta scores file missing: {synthetic_hybrid_meta_fname}");
        
        if not required_files_exist and not (synthetic_model_specific_output_dir / synthetic_uncertainty_fname).exists():
             logging.error(f"Core uncertainty file for synthetic AUROC calc missing in {synthetic_model_specific_output_dir}.")
             sys.exit(1)

        synthetic_gen_pkl_for_auroc = synthetic_model_specific_output_dir / synthetic_gen_pkl_fname
        if not synthetic_gen_pkl_for_auroc.exists():
            logging.error(f"Synthetic generations PKL {synthetic_gen_pkl_for_auroc} missing for AUROC calculation.")

        stage_args = construct_args(config, calc_auroc_args_map)
        positional_args_auroc = [f"{run_id}_synth_auroc", str(synthetic_model_specific_output_dir)]
        stage_args.extend(["--output_json", str(synthetic_model_specific_output_dir / synthetic_auroc_json_fname)])
        stage_args.extend([
            "--uncertainty_pkl_filename_override", synthetic_uncertainty_fname,
            "--is_scores_csv_filename_override", synthetic_is_scores_fname,
            "--hybrid_meta_csv_filename_override", synthetic_hybrid_meta_fname,
            "--generations_pkl_filename_override", synthetic_gen_pkl_fname
        ])
        if "analysis_save_aurac_plots_synthetic" in config:
            stage_args = [arg for arg in stage_args if arg != '--save_plots']
            if config["analysis_save_aurac_plots_synthetic"]:
                stage_args.append("--save_plots")
        
        success &= run_stage("calculate_aurocs_on_synthetic", "src.analysis.calculate_full_set_aurocs", config, positional_args_auroc + stage_args)
        if not success:
            sys.exit(1)

    qa_run_output_dir = current_run_main_output_dir 
    if not is_any_synthetic_stage_active and "base_output_dir_qa" in config :
        qa_run_output_dir = get_run_output_dir(Path(config["base_output_dir_qa"]), run_id)
    elif not is_any_synthetic_stage_active and "base_output_dir" in config :
        qa_run_output_dir = get_run_output_dir(Path(config["base_output_dir"]), run_id)


    if run_flags.get("generate", False):
        stage_args = construct_args(config, gen_args_map)
        stage_args.extend(["--output_dir", str(qa_run_output_dir)])
        success &= run_stage("generate", "src.semantic_uncertainty.generate", config, stage_args)
        if not success:
            sys.exit(1)

    if run_flags.get("compute_se", False):
        gen_output_pkl = qa_run_output_dir / "validation_generations.pkl"
        if gen_output_pkl.exists():
            stage_args = construct_args(config, se_args_map)
            stage_args.extend(["--run_dir", str(qa_run_output_dir)])
            success &= run_stage("compute_se", "src.semantic_uncertainty.compute_se", config, stage_args)
            if not success:
                sys.exit(1)
        else:
            logging.warning(f"QA Generate output {gen_output_pkl} missing. Skipping QA compute_se.")

    if run_flags.get("probe_is", False):
        gen_output_pkl = qa_run_output_dir / "validation_generations.pkl"
        if gen_output_pkl.exists():
             stage_args = construct_args(config, is_args_map)
             positional_args = [run_id] 
             stage_args.extend(["--base_dir", str(qa_run_output_dir)])
             success &= run_stage("probe_is", "src.internal_signals.probe", config, positional_args + stage_args)
             if not success:
                 sys.exit(1)
        else:
            logging.warning(f"Input {gen_output_pkl} missing for QA probe_is. Skipping.")

    if run_flags.get("train_hybrid_meta", False):
        uncertainty_pkl = qa_run_output_dir / "uncertainty_measures.pkl"
        is_scores_all_csv = qa_run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
        if uncertainty_pkl.exists() and is_scores_all_csv.exists():
             stage_args = construct_args(config, hybrid_meta_train_args_map)
             positional_args = [run_id, str(qa_run_output_dir)]
             success &= run_stage("train_hybrid_meta", "src.hybrid.train_hybrid_meta", config, positional_args + stage_args)
             if not success:
                 sys.exit(1)
        else:
            logging.warning(f"Inputs missing for QA train_hybrid_meta in {qa_run_output_dir}. Skipping.")

    if run_flags.get("calculate_aurocs", False):
         uncertainty_pkl = qa_run_output_dir / "uncertainty_measures.pkl"
         is_scores_all_csv = qa_run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
         hybrid_meta_scores_all_csv = qa_run_output_dir / f"{run_id}_hybrid_meta_scores_all.csv"
         if uncertainty_pkl.exists() and is_scores_all_csv.exists() and hybrid_meta_scores_all_csv.exists():
             stage_args = construct_args(config, calc_auroc_args_map)
             positional_args = [run_id, str(qa_run_output_dir)]
             auroc_json_output = qa_run_output_dir / config.get("metrics_output_json", f"{run_id}_full_set_metrics.json")
             stage_args.extend(["--output_json", str(auroc_json_output)])
             success &= run_stage("calculate_aurocs", "src.analysis.calculate_full_set_aurocs", config, positional_args + stage_args)
         else:
             logging.warning(f"Inputs missing for QA calculate_aurocs in {qa_run_output_dir}. Skipping.")

    if run_flags.get("prepare_analysis", False):
        anno_dir = Path(config.get("processed_annotation_dir", "annotation_data"))
        reviewed_anno_file_name = config.get("annotation_csv_filename", f"{run_id}_annotations_processed.csv")
        reviewed_anno_file = anno_dir / run_id / reviewed_anno_file_name
        
        is_scores_all_csv = qa_run_output_dir / f"{run_id}_internal_signal_scores_all.csv"
        hybrid_meta_scores_all_csv = qa_run_output_dir / f"{run_id}_hybrid_meta_scores_all.csv"
        uncertainty_pkl = qa_run_output_dir / "uncertainty_measures.pkl"
        generations_pkl = qa_run_output_dir / "validation_generations.pkl"
        final_output_file = qa_run_output_dir / f"{run_id}_final_analysis_data.csv"

        inputs_ok = True
        required_files = {"Reviewed Annotations": reviewed_anno_file, "IS Scores (All)": is_scores_all_csv,
                          "Hybrid Meta Scores (All)": hybrid_meta_scores_all_csv, "Uncertainty PKL": uncertainty_pkl,
                          "Generations PKL": generations_pkl }
        for name, path_obj in required_files.items():
            if not path_obj.exists():
                logging.error(f"{name} not found: {path_obj} (for QA prepare_analysis).")
                inputs_ok = False
        
        if inputs_ok:
            positional_args = [str(reviewed_anno_file), str(qa_run_output_dir), str(final_output_file)]
            keyword_args = [
                "--is_scores_all_csv", str(is_scores_all_csv),
                "--hybrid_meta_scores_all_csv", str(hybrid_meta_scores_all_csv)
            ]
            keyword_args_from_config = construct_args(config, prep_args_map)
            full_args = positional_args + keyword_args + keyword_args_from_config
            success &= run_stage("prepare_analysis", "src.preparation.create_analysis_dataset", config, full_args)
            if not success:
                sys.exit(1)
        else:
            logging.error("Skipping QA prepare_analysis due to missing inputs.")

    if run_flags.get("evaluate_subtypes", False):
        analysis_input_file = qa_run_output_dir / f"{run_id}_final_analysis_data.csv"
        plot_dir_suffix = config.get("analysis_plot_dir_suffix", "subtype_plots")
        plot_output_dir_abs = qa_run_output_dir / plot_dir_suffix
        if analysis_input_file.exists():
             stage_args = construct_args(config, eval_args_map)
             positional_args = [str(analysis_input_file), run_id]
             stage_args.extend(["--plot_dir", str(plot_output_dir_abs)])
             full_args = positional_args + stage_args
             success &= run_stage("evaluate_subtypes", "src.analysis.evaluate_subtypes", config, full_args)
        else:
            logging.warning(f"Input file {analysis_input_file} missing for QA evaluate_subtypes. Skipping.")

    if run_flags.get("explain_detector", False):
        meta_model_filename = config.get("hybrid_meta_model_filename_override", f"{run_id}_hybrid_meta_model.pkl")
        analysis_csv_filename = config.get("final_analysis_data_csv_override", f"{run_id}_final_analysis_data.csv")
        scalers_filename = config.get("hybrid_meta_scalers_filename_override", f"{run_id}_hybrid_meta_scalers.pkl")

        meta_model_path = qa_run_output_dir / meta_model_filename
        analysis_csv_path = qa_run_output_dir / analysis_csv_filename
        scalers_path = qa_run_output_dir / scalers_filename

        if meta_model_path.exists() and analysis_csv_path.exists() and scalers_path.exists():
            positional_args_explain = [run_id, str(qa_run_output_dir)]
            
            script_args_explain = [
                "--meta_model_filename", meta_model_filename,
                "--final_analysis_data_csv", analysis_csv_filename,
                "--scalers_filename", scalers_filename
            ]
            if "xai_num_shap_summary_samples" in config:
                script_args_explain.extend(["--num_shap_summary_samples", str(config["xai_num_shap_summary_samples"])])
            if "xai_num_instance_plots" in config:
                script_args_explain.extend(["--num_instance_plots", str(config["xai_num_instance_plots"])])
            
            success &= run_stage("explain_detector", "src.analysis.explain_meta_learner", config, positional_args_explain + script_args_explain)
        else:
            logging.warning(f"Inputs for explain_detector stage not found. Missing one of: "
                            f"{meta_model_path}, {analysis_csv_path}, {scalers_path}. Skipping.")


    logging.info(f"\n--- Experiment Pipeline for Run {run_id} Finished ---")
    if not success: 
        logging.error("One or more critical stages failed during the run.")
        sys.exit(1)
    else:
        logging.info("All requested stages completed.")

if __name__ == "__main__":
    main()