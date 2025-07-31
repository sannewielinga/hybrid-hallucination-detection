import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import json
import matplotlib.pyplot as plt
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle
from src.utils.eval_utils import aurac, rejection_accuracy_curve, calculate_ece

DEFAULT_UNCERTAINTY_PKL = "uncertainty_measures.pkl"
DEFAULT_GENERATIONS_PKL = "validation_generations.pkl"
DEFAULT_IS_SCORES_CSV_TEMPLATE = "{run_id}_internal_signal_scores_all.csv"
DEFAULT_HYBRID_META_CSV_TEMPLATE = "{run_id}_hybrid_meta_scores_all.csv"
DEFAULT_METRICS_JSON_TEMPLATE = "{run_id}_full_set_metrics.json"

def save_plot(figure, filepath):
    """
    Saves a matplotlib figure to the specified file path and logs a success message.
    If an exception occurs, logs an error message. Closes the figure in all cases to prevent memory leaks.
    """
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filepath, bbox_inches="tight", dpi=150)
        logging.info(f"Plot saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filepath}: {e}")
    finally:
        plt.close(figure)

def calculate_full_set_metrics(run_id_arg, run_dir_str, output_json_path_override=None, save_plots_flag=True, n_ece_bins_arg=10, n_splits_cv_arg=5, random_seed_cv_arg=42,
                               uncertainty_pkl_filename_override=None,
                               is_scores_csv_filename_override=None,
                               hybrid_meta_csv_filename_override=None,
                               generations_pkl_filename_override=None
                               ):
    """
    Calculate various metrics for a given run, such as AUROC, AUPRC, AURAC, and ECE, based on uncertainty,
    internal signal, hybrid, and other scores. Optionally save plots for the metrics.

    Args:
        run_id_arg (str): The run identifier.
        run_dir_str (str): Directory path where run data is stored.
        output_json_path_override (str, optional): Path to save computed metrics as JSON. Defaults to None.
        save_plots_flag (bool, optional): Flag to save plots of metrics. Defaults to True.
        n_ece_bins_arg (int, optional): Number of bins for Expected Calibration Error (ECE) calculation. Defaults to 10.
        n_splits_cv_arg (int, optional): Number of splits for cross-validation. Defaults to 5.
        random_seed_cv_arg (int, optional): Random seed for cross-validation. Defaults to 42.
        uncertainty_pkl_filename_override (str, optional): Override filename for uncertainty measures PKL file. Defaults to None.
        is_scores_csv_filename_override (str, optional): Override filename for internal signal scores CSV file. Defaults to None.
        hybrid_meta_csv_filename_override (str, optional): Override filename for hybrid meta scores CSV file. Defaults to None.
        generations_pkl_filename_override (str, optional): Override filename for generations PKL file. Defaults to None.

    Returns:
        dict: A dictionary containing calculated metrics for each method.
    """

    setup_logger()
    run_dir = Path(run_dir_str)
    logging.info(f"--- Calculating Full-Set Metrics for Run: {run_id_arg} in dir {run_dir} ---")

    unc_fname = uncertainty_pkl_filename_override if uncertainty_pkl_filename_override else DEFAULT_UNCERTAINTY_PKL
    is_fname = is_scores_csv_filename_override if is_scores_csv_filename_override else DEFAULT_IS_SCORES_CSV_TEMPLATE.format(run_id=run_id_arg)
    hybrid_meta_fname = hybrid_meta_csv_filename_override if hybrid_meta_csv_filename_override else DEFAULT_HYBRID_META_CSV_TEMPLATE.format(run_id=run_id_arg)
    gen_fname = generations_pkl_filename_override if generations_pkl_filename_override else DEFAULT_GENERATIONS_PKL

    uncertainty_path = run_dir / unc_fname
    is_scores_path = run_dir / is_fname
    hybrid_meta_path = run_dir / hybrid_meta_fname
    generations_path = run_dir / gen_fname


    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path) 
    
    df_is = None
    if is_scores_path.is_file():
        df_is = pd.read_csv(is_scores_path); df_is['id'] = df_is['id'].astype(str).str.strip()
    else:
        logging.warning(f"IS scores file missing: {is_scores_path}. IS Probe method will be skipped.")
    
    df_hybrid_meta = None
    if hybrid_meta_path.is_file():
        df_hybrid_meta = pd.read_csv(hybrid_meta_path); df_hybrid_meta['id'] = df_hybrid_meta['id'].astype(str).str.strip()
    else:
         logging.warning(f"Hybrid meta scores file missing: {hybrid_meta_path}. Hybrid (Meta) method will be skipped.")

    if uncertainty_data is None:
        logging.error(f"Uncertainty data not found at {uncertainty_path}. Cannot proceed.")
        return None
    if generations_data is None:
        logging.warning(f"Generations data not found at {generations_path}. Hybrid (Simple Avg) may be affected or task ID alignment might fail.")

    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        validation_is_false_raw = uncertainty_data.get("validation_is_false")
        
        if "task_ids" in uncertainty_data and isinstance(uncertainty_data["task_ids"], list):
            task_ids_ordered = [str(tid).strip() for tid in uncertainty_data["task_ids"]]
        elif generations_data:
            task_ids_ordered = [str(tid).strip() for tid in generations_data.keys()]
            logging.warning("Using task_ids from generations_data as not found in uncertainty_data.")
        else:
            num_samples_from_false = len(validation_is_false_raw) if validation_is_false_raw is not None else 0
            task_ids_ordered = [f"sample_{i}" for i in range(num_samples_from_false)]
            logging.error("Neither uncertainty_data nor generations_data provided task_ids. Using generated sequential IDs. Alignment may be incorrect.")


        num_tasks = len(task_ids_ordered)
        if validation_is_false_raw is None:
            logging.error("'validation_is_false' missing from uncertainty data. Cannot calculate metrics.")
            return None

        def get_data_aligned(key, source_dict, target_len, default_val_if_missing=np.nan):
            data = source_dict.get(key)
            if data is None:
                logging.warning(f"Key '{key}' not found in source dictionary (measures).")
                return [default_val_if_missing] * target_len
            current_len = len(data)
            data_list = list(data)
            if current_len < target_len:
                logging.warning(f"Padding data for '{key}'. Expected {target_len}, got {current_len}.")
                return data_list + [default_val_if_missing] * (target_len - current_len)
            elif current_len > target_len:
                logging.warning(f"Truncating data for '{key}'. Expected {target_len}, got {current_len}.")
                return data_list[:target_len]
            return data_list

        validation_is_false_processed = get_data_aligned("validation_is_false", uncertainty_data, num_tasks, default_val_if_missing=True)
        ground_truth_incorrect = np.array(validation_is_false_processed, dtype=object)
        ground_truth_correct = np.array([np.nan if pd.isna(x) else not x for x in validation_is_false_processed], dtype=object)

        scores_dict = {'id': task_ids_ordered}
        scores_dict['semantic_entropy'] = get_data_aligned('semantic_entropy', measures, num_tasks)
        scores_dict['normalized_semantic_entropy'] = get_data_aligned('normalized_semantic_entropy', measures, num_tasks)
        scores_dict['naive_entropy'] = get_data_aligned('regular_entropy', measures, num_tasks)

        p_true_score_key = None
        if 'p_false_fixed' in measures:
            p_false_scores = get_data_aligned('p_false_fixed', measures, num_tasks)
            scores_dict['p_true_score'] = [1.0 - p if not pd.isna(p) else np.nan for p in p_false_scores]
            p_true_score_key = 'p_true_score'
        elif 'p_true_logprob' in measures:
            p_true_logprob = get_data_aligned('p_true_logprob', measures, num_tasks)
            scores_dict['p_true_score'] = [np.exp(p) if not pd.isna(p) else np.nan for p in p_true_logprob]
            p_true_score_key = 'p_true_score'
        
        df_main = pd.DataFrame(scores_dict)

    except Exception as e:
        logging.error(f"Error preparing main data from PKLs: {e}", exc_info=True)
        return None

    merged_df = df_main
    if df_is is not None:
        merged_df = pd.merge(merged_df, df_is, on='id', how='left')
    if df_hybrid_meta is not None:
        merged_df = pd.merge(merged_df, df_hybrid_meta, on='id', how='left')

    merged_df['gt_incorrect'] = ground_truth_incorrect
    merged_df['gt_correct'] = ground_truth_correct

    hybrid_simple_col = 'hybrid_simple_score'
    merged_df[hybrid_simple_col] = np.nan 

    if 'semantic_entropy' in merged_df.columns and 'internal_signal_score' in merged_df.columns and generations_data is not None:
        logging.info(f"Calculating OOF Hybrid (Simple Avg) score using {n_splits_cv_arg}-Fold CV...")
        df_for_cv = merged_df.dropna(subset=['semantic_entropy', 'internal_signal_score', 'gt_incorrect']).copy()
        if len(df_for_cv) >= n_splits_cv_arg and len(df_for_cv['gt_incorrect'].unique()) > 1:
            X_se = df_for_cv['semantic_entropy'].values
            X_is = df_for_cv['internal_signal_score'].values
            y_cv = df_for_cv['gt_incorrect'].astype(bool).values
            ids_cv = df_for_cv['id'].values

            skf = StratifiedKFold(n_splits=n_splits_cv_arg, shuffle=True, random_state=random_seed_cv_arg)
            oof_hybrid_simple_scores = np.full(len(y_cv), np.nan)
            
            original_indices_map = {original_id: i for i, original_id in enumerate(merged_df['id'])}
            
            for fold_idx, (train_index_cv, test_index_cv) in enumerate(skf.split(np.zeros_like(y_cv), y_cv)):
                X_se_train_fold, X_se_test_fold = X_se[train_index_cv], X_se[test_index_cv]
                X_is_train_fold, X_is_test_fold = X_is[train_index_cv], X_is[test_index_cv]
                y_train_fold_cv = y_cv[train_index_cv]

                if len(np.unique(y_train_fold_cv)) < 2:
                    logging.warning(f"Fold {fold_idx + 1} (SimpleAvg CV): Train split has only one class. Skipping fold scaling, may impact results.")
                    scaler_se_fold = MinMaxScaler().fit(X_se.reshape(-1,1)) if len(X_se)>0 else None
                    scaler_is_fold = MinMaxScaler().fit(X_is.reshape(-1,1)) if len(X_is)>0 else None

                else:
                    scaler_se_fold = MinMaxScaler()
                    scaler_is_fold = MinMaxScaler()
                    try:
                        scaler_se_fold.fit(X_se_train_fold.reshape(-1, 1))
                    except ValueError:
                        scaler_se_fold = None
                        logging.warning("SE constant in fold train.")
                    try:
                        scaler_is_fold.fit(X_is_train_fold.reshape(-1, 1))
                    except ValueError:
                        scaler_is_fold = None
                        logging.warning("IS constant in fold train.")


                X_se_test_norm_fold = scaler_se_fold.transform(X_se_test_fold.reshape(-1, 1)).flatten() if scaler_se_fold else np.full_like(X_se_test_fold, 0.5)
                X_is_test_norm_fold = scaler_is_fold.transform(X_is_test_fold.reshape(-1, 1)).flatten() if scaler_is_fold else np.full_like(X_is_test_fold, 0.5)
                
                current_fold_hybrid_scores = 0.5 * X_se_test_norm_fold + 0.5 * X_is_test_norm_fold
                
                test_ids_this_fold = ids_cv[test_index_cv]
                for original_id, score in zip(test_ids_this_fold, current_fold_hybrid_scores):
                    if original_id in original_indices_map:
                        merged_df.loc[original_indices_map[original_id], hybrid_simple_col] = score
            logging.info(f"Calculated OOF {hybrid_simple_col} for up to {len(df_for_cv)} samples.")
        else:
             logging.warning("Not enough data or classes for Hybrid (Simple Avg) K-Fold CV. Scores will be NaN.")
    else:
        logging.warning("Cannot calculate Hybrid (Simple Avg): 'semantic_entropy', 'internal_signal_score', or generations_data (for task_ids) missing.")


    plot_dir_path = None
    if save_plots_flag:
        plot_dir_path = run_dir / "aurac_plots"; plot_dir_path.mkdir(parents=True, exist_ok=True)

    results = {}
    methods_to_calc = {
        'SE': 'semantic_entropy', 'NSE': 'normalized_semantic_entropy', 'Naive': 'naive_entropy',
        'IS Probe': 'internal_signal_score', 'Hybrid (Simple Avg)': hybrid_simple_col, 'Hybrid (Meta)': 'hybrid_meta_score'
    }
    if p_true_score_key:
        methods_to_calc['P(True)'] = p_true_score_key

    logging.info("Calculating AUROC, AUPRC, AURAC, and ECE...")
    for method_name, col_name in methods_to_calc.items():
        results[method_name] = {'AUROC': None, 'AUPRC': None, 'AURAC': None, 'ECE': None}
        if col_name not in merged_df.columns or merged_df[col_name].isnull().all():
            logging.warning(f"Column '{col_name}' for method '{method_name}' missing or all NaN. Skipping.")
            continue

        df_filtered = merged_df.dropna(subset=[col_name, 'gt_incorrect', 'gt_correct']).copy()
        if df_filtered.empty:
            logging.warning(f"No valid scores/labels overlap for '{method_name}'. Skipping.")
            continue

        y_true_incorrect_valid = df_filtered['gt_incorrect'].astype(float)
        y_true_correct_valid = df_filtered['gt_correct'].astype(float)
        y_score_raw_valid = df_filtered[col_name].astype(float).values

        unique_classes_incorrect = np.unique(y_true_incorrect_valid[~np.isnan(y_true_incorrect_valid)])
        auroc_val, auprc_val, aurac_val, ece_val = np.nan, np.nan, np.nan, np.nan

        if len(unique_classes_incorrect) < 2:
            logging.warning(f"Ground truth for '{method_name}' has only one class after filtering NaNs. Cannot calculate some metrics.")
        else:
            y_score_metric = y_score_raw_valid
            if method_name == 'P(True)':
                 y_score_metric = 1.0 - y_score_raw_valid

            try:
                current_auroc = roc_auc_score(y_true_incorrect_valid, y_score_metric)
                if current_auroc < 0.5 and method_name != 'P(True)':
                    y_score_metric = -y_score_metric 
                    current_auroc = roc_auc_score(y_true_incorrect_valid, y_score_metric)
                auroc_val = current_auroc
            except ValueError:
                pass
            except Exception as e:
                logging.error(f"AUROC error for {method_name}: {e}")
            if not pd.isna(auroc_val):
                logging.info(f"  AUROC {method_name}: {auroc_val:.4f}")

            try:
                precision, recall, _ = precision_recall_curve(y_true_incorrect_valid, y_score_metric)
                if precision is not None and recall is not None and len(precision)>1 and len(recall)>1:
                    auprc_val = auc(recall, precision)
            except ValueError:
                pass
            except Exception as e:
                logging.error(f"AUPRC error for {method_name}: {e}")
            if not pd.isna(auprc_val):
                logging.info(f"  AUPRC {method_name}: {auprc_val:.4f}")

            y_score_for_aurac = y_score_raw_valid
            if method_name == 'P(True)':
                y_score_for_aurac = 1.0 - y_score_raw_valid 
            elif not pd.isna(auroc_val) and auroc_val < 0.5:
                y_score_for_aurac = -y_score_raw_valid

            try:
                aurac_val = aurac(y_true_correct_valid, y_score_for_aurac)
            except Exception as e:
                logging.error(f"AURAC error for {method_name}: {e}")
            if not pd.isna(aurac_val):
                logging.info(f"  AURAC {method_name}: {aurac_val:.4f}")
            
            y_prob_for_ece = y_score_metric
            
            is_prob = np.all((y_prob_for_ece[~np.isnan(y_prob_for_ece)] >= 0) & (y_prob_for_ece[~np.isnan(y_prob_for_ece)] <= 1))
            if is_prob and method_name in ['IS Probe', 'Hybrid (Simple Avg)', 'Hybrid (Meta)', 'P(True)']:
                try:
                    ece_val = calculate_ece(y_true_incorrect_valid[~np.isnan(y_prob_for_ece)], y_prob_for_ece[~np.isnan(y_prob_for_ece)], n_bins=n_ece_bins_arg)
                except Exception as e:
                    logging.error(f"ECE error for {method_name}: {e}")
                if not pd.isna(ece_val):
                    logging.info(f"  ECE   {method_name}: {ece_val:.4f}")
            elif method_name in ['IS Probe', 'Hybrid (Simple Avg)', 'Hybrid (Meta)', 'P(True)']:
                logging.warning(f"Scores for {method_name} not in [0,1] for ECE. Min: {np.nanmin(y_prob_for_ece):.2f}, Max: {np.nanmax(y_prob_for_ece):.2f}.")


            if save_plots_flag and plot_dir_path and not pd.isna(aurac_val):
                 try:
                    fractions, accuracies = rejection_accuracy_curve(y_true_correct_valid, y_score_for_aurac)
                    fig_rej, ax_rej = plt.subplots(figsize=(8, 6))
                    plot_mask = ~np.isnan(accuracies)
                    if np.any(plot_mask):
                        ax_rej.plot(fractions[plot_mask], accuracies[plot_mask], marker='o', linestyle='-')
                        ax_rej.set_title(f'Rejection Accuracy Curve for {method_name} (Run: {run_id_arg})')
                        ax_rej.set_xlabel('Fraction of Samples Rejected (Highest Uncertainty First)')
                        ax_rej.set_ylabel('Accuracy of Accepted Samples')
                        ax_rej.grid(True, linestyle='--')
                        min_acc_plot = np.nanmin(accuracies[plot_mask]) if np.any(~np.isnan(accuracies[plot_mask])) else 0.5
                        ax_rej.set_ylim(bottom=min(0.5, min_acc_plot - 0.05), top=1.05)
                        overall_acc_val = np.mean(y_true_correct_valid[~np.isnan(y_true_correct_valid)])
                        ax_rej.axhline(y=overall_acc_val, color='r', linestyle='--', label=f'Overall Acc: {overall_acc_val:.3f}')
                        ax_rej.legend(); ax_rej.text(0.05, 0.1, f'AURAC = {aurac_val:.4f}', transform=ax_rej.transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                        plot_fname = plot_dir_path / f"{run_id_arg}_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_RejectionCurve.png"
                        save_plot(fig_rej, plot_fname)
                    else: plt.close(fig_rej)
                 except Exception as plot_e:
                    logging.error(f"Failed to generate/save rejection plot for {method_name}: {plot_e}")
                    if 'fig_rej' in locals(): plt.close(fig_rej)
        
        results[method_name]['AUROC'] = auroc_val
        results[method_name]['AUPRC'] = auprc_val
        results[method_name]['AURAC'] = aurac_val
        results[method_name]['ECE'] = ece_val

    logging.info("\n--- Final Full-Set Metrics Results ---")
    print(json.dumps(results, indent=2))
    
    output_json_final_path = Path(output_json_path_override) if output_json_path_override else run_dir / DEFAULT_METRICS_JSON_TEMPLATE.format(run_id=run_id_arg)
    try:
        output_json_final_path.parent.mkdir(parents=True, exist_ok=True)
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj) if not np.isnan(obj) else None
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if pd.isna(obj):
                    return None
                return super(NpEncoder, self).default(obj)
        with open(output_json_final_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        logging.info(f"Saved results to: {output_json_final_path}")
    except Exception as e:
        logging.error(f"Failed to save results JSON: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate full-set metrics with OOF Simple Hybrid.")
    parser.add_argument("run_id", type=str)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--uncertainty_pkl_filename_override", default=None)
    parser.add_argument("--is_scores_csv_filename_override", default=None)
    parser.add_argument("--hybrid_meta_csv_filename_override", default=None)
    parser.add_argument("--generations_pkl_filename_override", default=None)


    args = parser.parse_args()

    calculate_full_set_metrics(
        args.run_id, args.run_dir, args.output_json, args.save_plots,
        n_ece_bins_arg=args.ece_bins, n_splits_cv_arg=args.n_splits, random_seed_cv_arg=args.seed,
        uncertainty_pkl_filename_override=args.uncertainty_pkl_filename_override,
        is_scores_csv_filename_override=args.is_scores_csv_filename_override,
        hybrid_meta_csv_filename_override=args.hybrid_meta_csv_filename_override,
        generations_pkl_filename_override=args.generations_pkl_filename_override
    )