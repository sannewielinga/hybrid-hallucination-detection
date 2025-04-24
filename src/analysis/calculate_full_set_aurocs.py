import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import json
import matplotlib.pyplot as plt
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle
from src.utils.eval_utils import aurac, rejection_accuracy_curve


def calculate_full_set_metrics(run_id, run_dir, output_json_path=None, save_plots=True):
    setup_logger()
    run_dir = Path(run_dir)
    logging.info(f"--- Calculating Full-Set AUROC, AUPRC & AURAC for Run: {run_id} ---")

    uncertainty_path = run_dir / "uncertainty_measures.pkl"
    generations_path = run_dir / "validation_generations.pkl"
    is_scores_path = run_dir / f"{run_id}_internal_signal_scores_all.csv"
    hybrid_meta_path = run_dir / f"{run_id}_hybrid_meta_scores_all.csv"

    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)
    df_is = None
    df_hybrid_meta = None

    if is_scores_path.is_file():
        df_is = pd.read_csv(is_scores_path)
        df_is['id'] = df_is['id'].astype(str).str.strip()
    else:
        logging.warning(f"IS scores file missing: {is_scores_path}")
    if hybrid_meta_path.is_file():
        df_hybrid_meta = pd.read_csv(hybrid_meta_path)
        df_hybrid_meta['id'] = df_hybrid_meta['id'].astype(str).str.strip()
    else:
         logging.warning(f"Hybrid meta scores file missing: {hybrid_meta_path}")

    if uncertainty_data is None or generations_data is None:
        return None

    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        validation_is_false_raw = uncertainty_data.get("validation_is_false")
        task_ids_ordered = list(generations_data.keys())
        num_tasks = len(task_ids_ordered)
        if validation_is_false_raw is None:
            logging.error("'validation_is_false' missing.")
            return None

        def get_data_aligned(key, source_dict, target_len):
            data = source_dict.get(key)
            if data is None:
                logging.warning(f"Key '{key}' not found in source dictionary.")
                return [np.nan] * target_len
            current_len = len(data)
            if current_len < target_len:
                logging.warning(f"Padding data for '{key}'. Expected {target_len}, got {current_len}.")
                return data + [np.nan] * (target_len - current_len)
            elif current_len > target_len:
                logging.warning(f"Truncating data for '{key}'. Expected {target_len}, got {current_len}.")
                return data[:target_len]
            return data

        validation_is_false_processed = get_data_aligned("validation_is_false", uncertainty_data, num_tasks)
        ground_truth_incorrect = np.array(validation_is_false_processed, dtype=object)
        ground_truth_correct = np.array([
            np.nan if pd.isna(x) else not x
            for x in validation_is_false_processed
        ], dtype=object)


        scores_dict = {'id': [str(tid).strip() for tid in task_ids_ordered]}
        scores_dict['semantic_entropy'] = get_data_aligned('semantic_entropy', measures, num_tasks)
        scores_dict['naive_entropy'] = get_data_aligned('regular_entropy', measures, num_tasks)

        p_true_key_for_metrics = None
        if 'p_false_fixed' in measures:
            scores_dict['p_true_metric'] = get_data_aligned('p_false_fixed', measures, num_tasks)
            p_true_key_for_metrics = 'p_true_metric'
        elif 'p_true_logprob' in measures:
            p_true_logprob = get_data_aligned('p_true_logprob', measures, num_tasks)
            scores_dict['p_true_metric'] = [1.0 - np.exp(p) if not pd.isna(p) else np.nan for p in p_true_logprob]
            p_true_key_for_metrics = 'p_true_metric'
        if not p_true_key_for_metrics: logging.warning("No P(True) data found for metrics.")

        df_main = pd.DataFrame(scores_dict)

    except Exception as e:
        logging.error(f"Error preparing data: {e}", exc_info=True)
        return None

    merged_df = df_main
    if df_is is not None: merged_df = pd.merge(merged_df, df_is, on='id', how='left')
    if df_hybrid_meta is not None: merged_df = pd.merge(merged_df, df_hybrid_meta, on='id', how='left')

    merged_df['gt_incorrect'] = ground_truth_incorrect
    merged_df['gt_correct'] = ground_truth_correct


    se_col, is_col, hybrid_simple_col = 'semantic_entropy', 'internal_signal_score', 'hybrid_simple_score'
    if se_col in merged_df.columns and is_col in merged_df.columns:
        logging.info("Calculating simple hybrid score for full set...")
        scaler_se, scaler_is = MinMaxScaler(), MinMaxScaler()
        se_norm_col, is_norm_col = f"{se_col}_norm", f"{is_col}_norm"
        merged_df[se_norm_col], merged_df[is_norm_col] = np.nan, np.nan
        se_valid_mask = merged_df[se_col].notna(); is_valid_mask = merged_df[is_col].notna()
        if se_valid_mask.any():
            try: merged_df.loc[se_valid_mask, se_norm_col] = scaler_se.fit_transform(merged_df.loc[se_valid_mask, [se_col]])
            except ValueError: merged_df.loc[se_valid_mask, se_norm_col] = 0.5
        if is_valid_mask.any():
            try: merged_df.loc[is_valid_mask, is_norm_col] = scaler_is.fit_transform(merged_df.loc[is_valid_mask, [is_col]])
            except ValueError: merged_df.loc[is_valid_mask, is_norm_col] = 0.5
        merged_df[hybrid_simple_col] = 0.5 * merged_df[se_norm_col] + 0.5 * merged_df[is_norm_col]
    else: merged_df[hybrid_simple_col] = np.nan

    plot_dir = None
    if save_plots:
        plot_dir = Path(run_dir) / "aurac_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving rejection curve plots to: {plot_dir}")


    results = {}
    methods_to_calc = {
        'SE': 'semantic_entropy',
        'Naive': 'naive_entropy',
        'IS Probe': 'internal_signal_score',
        'Hybrid (Simple Avg)': hybrid_simple_col,
        'Hybrid (Meta)': 'hybrid_meta_score'
    }
    if p_true_key_for_metrics: methods_to_calc['P(True)'] = p_true_key_for_metrics

    logging.info("Calculating AUROC, AUPRC, and AURAC against automated labels (Full Validation Set)...")
    for method_name, col_name in methods_to_calc.items():
        results[method_name] = {'AUROC': None, 'AUPRC': None, 'AURAC': None}

        if col_name not in merged_df.columns:
            logging.warning(f"Column '{col_name}' for method '{method_name}' missing. Skipping.")
            continue

        df_filtered = merged_df.dropna(subset=[col_name, 'gt_incorrect', 'gt_correct']).copy()

        if df_filtered.empty:
            logging.warning(f"No valid scores/labels overlap for '{method_name}' after NaN check. Skipping.")
            continue

        y_true_incorrect_valid = df_filtered['gt_incorrect'].astype(bool).values
        y_true_correct_valid = df_filtered['gt_correct'].astype(bool).values
        y_score_uncertainty_valid = df_filtered[col_name].values

        unique_classes_incorrect = np.unique(y_true_incorrect_valid)
        if len(unique_classes_incorrect) < 2:
            logging.warning(f"Ground truth for '{method_name}' has only one class ({unique_classes_incorrect[0]}) after filtering. Cannot calculate metrics reliably.")
            auroc_val, auprc_val, aurac_val = np.nan, np.nan, np.nan
        else:
            try: auroc_val = roc_auc_score(y_true_incorrect_valid, y_score_uncertainty_valid)
            except Exception as e: logging.error(f"AUROC calc error for {method_name}: {e}"); auroc_val = np.nan
            logging.info(f"  AUROC {method_name}: {auroc_val:.4f}")

            try:
                precision, recall, _ = precision_recall_curve(y_true_incorrect_valid, y_score_uncertainty_valid)
                if precision is not None and recall is not None and len(precision) > 1 and len(recall) > 1:
                    auprc_val = auc(recall, precision)
                else:
                    logging.warning(f"Invalid precision/recall arrays for AUPRC {method_name}. Setting AUPRC to NaN.")
                    auprc_val = np.nan
            except Exception as e: logging.error(f"AUPRC calc error for {method_name}: {e}"); auprc_val = np.nan
            logging.info(f"  AUPRC {method_name}: {auprc_val:.4f}")

            try:
                aurac_val = aurac(y_true_correct_valid, y_score_uncertainty_valid)
            except Exception as e: logging.error(f"AURAC calc error for {method_name}: {e}"); aurac_val = np.nan
            logging.info(f"  AURAC {method_name}: {aurac_val:.4f}")

            if save_plots and plot_dir and not np.isnan(aurac_val):
                 try:
                    fractions, accuracies = rejection_accuracy_curve(
                        y_true_correct_valid, y_score_uncertainty_valid
                    )
                    fig_rej, ax_rej = plt.subplots(figsize=(8, 6))
                    plot_mask = ~np.isnan(accuracies)
                    if np.any(plot_mask):
                        ax_rej.plot(fractions[plot_mask], accuracies[plot_mask], marker='o', linestyle='-')
                        ax_rej.set_title(f'Rejection Accuracy Curve for {method_name} (Run: {run_id})')
                        ax_rej.set_xlabel('Fraction of Samples Rejected (Highest Uncertainty First)')
                        ax_rej.set_ylabel('Accuracy of Accepted Samples')
                        ax_rej.grid(True, linestyle='--')
                        min_acc_plot = np.nanmin(accuracies[plot_mask]) if np.any(~np.isnan(accuracies[plot_mask])) else 0.5
                        ax_rej.set_ylim(bottom=min(0.5, min_acc_plot - 0.05), top=1.05)
                        ax_rej.axhline(y=np.mean(y_true_correct_valid), color='r', linestyle='--', label=f'Overall Acc: {np.mean(y_true_correct_valid):.3f}')
                        ax_rej.legend()
                        ax_rej.text(0.05, 0.1, f'AURAC = {aurac_val:.4f}', transform=ax_rej.transAxes,
                                    bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

                        plot_filename_rej = plot_dir / f"{run_id}_{method_name}_RejectionCurve.png"
                        try:
                           save_plot(fig_rej, plot_filename_rej)
                        except NameError:
                           plt.savefig(plot_filename_rej, bbox_inches="tight", dpi=150)
                           logging.info(f"Plot saved to: {plot_filename_rej}")
                           plt.close(fig_rej)

                    else:
                        logging.warning(f"No valid accuracy points to plot rejection curve for {method_name}.")
                        plt.close(fig_rej)

                 except Exception as plot_e:
                    logging.error(f"Failed to generate/save rejection curve plot for {method_name}: {plot_e}")
                    if 'fig_rej' in locals(): plt.close(fig_rej)


        results[method_name]['AUROC'] = auroc_val if not pd.isna(auroc_val) else None
        results[method_name]['AUPRC'] = auprc_val if not pd.isna(auprc_val) else None
        results[method_name]['AURAC'] = aurac_val if not pd.isna(aurac_val) else None


    logging.info("\n--- Final Full-Set AUROC, AUPRC & AURAC Results ---")
    print(json.dumps(results, indent=2))
    if output_json_path:
        try:
            output_file = Path(output_json_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f: json.dump(results, f, indent=2)
            logging.info(f"Saved results to: {output_file}")
        except Exception as e: logging.error(f"Failed to save results JSON: {e}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate full-set AUROC, AUPRC, and AURAC.")
    parser.add_argument("run_id", type=str)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save results JSON.")
    parser.add_argument("--save_plots", action="store_true", help="Save rejection curve plots.")
    args = parser.parse_args()
    output_path = args.output_json or Path(args.run_dir) / f"{args.run_id}_full_set_metrics.json"
    calculate_full_set_metrics(args.run_id, args.run_dir, output_path, args.save_plots)