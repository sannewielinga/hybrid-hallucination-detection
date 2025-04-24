import argparse
import functools
import logging
import os
import pickle
import json
from pathlib import Path
import numpy as np

from src.utils import utils
from src.utils.eval_utils import (
    bootstrap,
    compatible_bootstrap,
    auroc,
    accuracy_at_quantile,
    area_under_thresholded_accuracy,
)

try:
    utils.setup_logger()
except AttributeError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

UNC_MEAS = "uncertainty_measures.pkl" 


def analyze_run(
    input_dir,
    answer_fractions_mode="default",
    output_json_path=None
):
    """
    Analyze the uncertainty measures from a run directory and compute performance metrics.

    This function will load the uncertainty measures from a specified directory and compute
    various performance metrics, including AUROC, accuracy at different answer fractions,
    and mean uncertainty. The results will be logged and optionally saved to a JSON file.

    Args:
        input_dir (str): Path to the directory containing the uncertainty measures (output of
            uncertainty_measures.py).
        answer_fractions_mode (str, optional): One of "default" (default answer fractions) or
            "finegrained" (answer fractions from 0 to 1 in increments of 0.05). Defaults to "default".
        output_json_path (str, optional): Path to save the analysis results to a JSON file. If not
            specified, results will only be logged.

    Returns:
        dict: A dictionary containing the analysis results, with keys "performance" and
            "uncertainty". The "performance" key contains a dictionary with keys for each
            performance metric, and the "uncertainty" key contains a dictionary with keys for
            each uncertainty measure and the corresponding performance metrics.
    """
    run_id = Path(input_dir).name
    logging.info(f"Analyzing run ID '{run_id}' from input directory: {input_dir}")

    if answer_fractions_mode == "default":
        answer_fractions = [0.8, 0.9, 0.95, 1.0]
    elif answer_fractions_mode == "finegrained":
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20 + 1)]
    else:
        logging.error(f"Invalid answer_fractions_mode: {answer_fractions_mode}")
        raise ValueError("Invalid answer_fractions_mode")

    rng = np.random.default_rng(41)
    eval_metrics = dict(
        zip(
            ["AUROC", "area_under_thresholded_accuracy", "mean_uncertainty"],
            list(
                zip(
                    [auroc, area_under_thresholded_accuracy, np.mean],
                    [compatible_bootstrap, compatible_bootstrap, bootstrap],
                )
            ),
        )
    )
    for answer_fraction in answer_fractions:
        key = f"accuracy_at_{answer_fraction}_answer_fraction"
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap,
        ]

    input_dir_path = Path(input_dir)
    uncertainty_file_path = input_dir_path / UNC_MEAS

    if not uncertainty_file_path.is_file():
        logging.error(f"Uncertainty file not found at: {uncertainty_file_path}")
        raise FileNotFoundError(f"Required file not found: {uncertainty_file_path}")

    logging.info(f"Loading uncertainty data from: {uncertainty_file_path}")
    try:
        with open(uncertainty_file_path, "rb") as file:
            results_old = pickle.load(file)
    except Exception as e:
        logging.error(f"Failed to load pickle file {uncertainty_file_path}: {e}")
        raise

    result_dict = {"performance": {}, "uncertainty": {}}

    if "validation_is_false" not in results_old:
        logging.error(f"'validation_is_false' key not found in {uncertainty_file_path}. Cannot compute performance.")
        return result_dict

    all_accuracies = dict()
    validation_is_false_gt = np.array(results_old["validation_is_false"])
    all_accuracies["accuracy"] = 1 - validation_is_false_gt

    for name, target in all_accuracies.items():
        result_dict["performance"][name] = {}
        mean_acc = np.mean(target)
        result_dict["performance"][name]["mean"] = mean_acc
        if np.var(target) > 0 and len(np.unique(target)) > 1:
             result_dict["performance"][name]["bootstrap"] = bootstrap(np.mean, rng)(target)
        else:
             result_dict["performance"][name]["bootstrap"] = {"std_err": 0, "low": mean_acc, "high": mean_acc}
             logging.warning(f"Skipping bootstrap for '{name}' due to constant values or single class.")

    rum = results_old.get("uncertainty_measures", {})
    if not rum:
        logging.warning(f"No 'uncertainty_measures' found in {uncertainty_file_path}. Skipping uncertainty analysis.")
        return result_dict

    if "p_false" in rum and "p_false_fixed" not in rum:
        logging.info("Calculating 'p_false_fixed' from 'p_false'.")
        p_false_fixed_list = []
        for x in rum["p_false"]:
            try:
                if x is not None and not np.isnan(x):
                    val = 1 - np.exp(1 - float(x))
                    p_false_fixed_list.append(val)
                else:
                    p_false_fixed_list.append(np.nan)
            except (TypeError, ValueError):
                p_false_fixed_list.append(np.nan)
        rum["p_false_fixed"] = p_false_fixed_list

    ground_truths = {"": validation_is_false_gt}
    if "validation_unanswerable" in results_old:
        ground_truths["_UNANSWERABLE"] = np.array(results_old["validation_unanswerable"])

    for measure_name, measure_values_raw in rum.items():
        logging.info(f"Computing metrics for uncertainty measure: '{measure_name}'")

        try:
            measure_values = np.array(measure_values_raw, dtype=float)
        except (ValueError, TypeError) as e:
            logging.warning(f"Could not convert measure values for '{measure_name}' to float array: {e}. Skipping this measure.")
            continue

        if np.isnan(measure_values).all():
            logging.warning(f"All values for '{measure_name}' are NaN. Skipping metric calculation.")
            continue

        for gt_suffix, validation_is_false in ground_truths.items():
            current_measure_name_log = measure_name + gt_suffix
            result_dict["uncertainty"][current_measure_name_log] = {}

            min_len = min(len(measure_values), len(validation_is_false))
            if len(measure_values) != len(validation_is_false):
                logging.warning(
                    f"Length mismatch for '{current_measure_name_log}'. "
                    f"Scores: {len(measure_values)}, Ground Truth: {len(validation_is_false)}. "
                    f"Using first {min_len} aligned samples for metric calculation."
                )
            aligned_measure_values = measure_values[:min_len]
            aligned_validation_is_false = validation_is_false[:min_len]
            aligned_validation_accuracy = 1 - aligned_validation_is_false

            valid_indices = ~np.isnan(aligned_measure_values)
            final_scores = aligned_measure_values[valid_indices]
            final_is_false = aligned_validation_is_false[valid_indices]
            final_accuracy = aligned_validation_accuracy[valid_indices]

            if len(final_scores) < 2 or len(np.unique(final_is_false)) < 2:
                 logging.warning(f"Not enough valid data points or only one class present for '{current_measure_name_log}' after NaN removal/alignment. Skipping metric calculations.")
                 continue

            fargs = {
                "AUROC": [final_is_false, final_scores],
                "area_under_thresholded_accuracy": [final_accuracy, final_scores],
                "mean_uncertainty": [final_scores],
            }

            for answer_fraction in answer_fractions:
                fargs[f"accuracy_at_{answer_fraction}_answer_fraction"] = [
                    final_accuracy,
                    final_scores,
                ]

            for fname, (function, bs_function) in eval_metrics.items():
                if fname != "mean_uncertainty" and len(np.unique(final_is_false)) < 2:
                    logging.warning(f"Skipping {fname} for '{current_measure_name_log}' due to single class ground truth.")
                    metric_i = np.nan
                    bs_results = {"std_err": np.nan, "low": np.nan, "high": np.nan}
                else:
                    try:
                        metric_i = function(*fargs[fname])

                        score_variance = np.var(fargs[fname][-1])
                        gt_variance = np.var(fargs[fname][0]) if len(fargs[fname]) > 1 else 1

                        can_bootstrap = score_variance > 0
                        if fname != "mean_uncertainty":
                             can_bootstrap &= (gt_variance > 0)

                        if can_bootstrap:
                            bs_results = bs_function(function, rng)(*fargs[fname])
                        else:
                            logging.warning(f"Skipping bootstrap for {fname} on '{current_measure_name_log}' due to zero variance in scores or ground truth.")
                            bs_results = {"std_err": 0, "low": metric_i, "high": metric_i}

                    except ValueError as ve:
                        logging.warning(f"Could not calculate {fname} for '{current_measure_name_log}': {ve}. Setting to NaN.")
                        metric_i = np.nan
                        bs_results = {"std_err": np.nan, "low": np.nan, "high": np.nan}
                    except Exception as e:
                        logging.error(f"Unexpected error calculating {fname} for '{current_measure_name_log}': {e}", exc_info=True)
                        metric_i = np.nan
                        bs_results = {"std_err": np.nan, "low": np.nan, "high": np.nan}


                result_dict["uncertainty"][current_measure_name_log][fname] = {}
                result_dict["uncertainty"][current_measure_name_log][fname]["mean"] = metric_i
                result_dict["uncertainty"][current_measure_name_log][fname]["bootstrap"] = bs_results
                logging.info(f"  Metric {fname}: {metric_i:.4f}")


    logging.info(f"\n--- Analysis Results for Run ID: {run_id} ---")
    try:
        results_json_str = json.dumps(result_dict, indent=2, default=lambda x: str(x) if isinstance(x, (np.ndarray, np.generic)) else x)
        logging.info(results_json_str)
    except TypeError as e:
        logging.error(f"Could not serialize results dictionary to JSON: {e}")
        logging.info(f"Raw results dictionary: {result_dict}")

    if output_json_path:
        output_file = Path(output_json_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_file, 'w') as f:
                f.write(results_json_str)
            logging.info(f"Saved analysis results to: {output_file}")
        except Exception as e:
            logging.error(f"Failed to save analysis results JSON to {output_file}: {e}")

    logging.info(f"--- Analysis finished for Run ID: {run_id} ---")
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute overall performance metrics from predicted uncertainties stored in a pkl file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the 'uncertainty_measures.pkl' file for the run.",
    )
    parser.add_argument(
        "--answer_fractions_mode",
         type=str,
         default="default",
         choices=["default", "finegrained"],
         help="Granularity for accuracy_at_quantile calculations."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the final results dictionary as a JSON file."
    )
    parser.add_argument(
        "--experiment_lot",
        type=str,
        default="Unnamed Analysis",
        help="A label for the analysis run (optional).",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Ignoring unknown args: {unknown}")

    logging.info(f"Starting analysis for directory: {args.input_dir}")
    try:
        analyze_run(
            input_dir=args.input_dir,
            answer_fractions_mode=args.answer_fractions_mode,
            output_json_path=args.output_json
        )
    except FileNotFoundError:
        logging.error(f"Analysis failed: Input file not found for directory {args.input_dir}")
    except Exception as e:
        logging.error(f"Analysis failed for directory {args.input_dir}: {e}", exc_info=True)

    logging.info("Analysis script finished.")