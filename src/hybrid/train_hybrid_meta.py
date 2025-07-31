import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle

DEFAULT_UNCERTAINTY_PKL = "uncertainty_measures.pkl"
DEFAULT_IS_SCORES_CSV = "_internal_signal_scores_all.csv" 
DEFAULT_GENERATIONS_PKL = "validation_generations.pkl" 
DEFAULT_HYBRID_META_CSV = "_hybrid_meta_scores_all.csv"
DEFAULT_MODEL_PKL = "_hybrid_meta_model.pkl"
DEFAULT_SCALERS_PKL = "_hybrid_meta_scalers.pkl"


def get_meta_classifier(classifier_name, random_seed):
    """
    Get a meta-classifier given the name and random seed.

    Args:
        classifier_name (str): The name of the meta-classifier. Choose 'logistic', 'svm', 'random_forest', or 'lightgbm'.
        random_seed (int): The random seed for the meta-classifier.

    Returns:
        A trained meta-classifier instance.
    """
    if classifier_name == 'logistic':
        logging.info("Using Logistic Regression meta-classifier.")
        return LogisticRegression(class_weight='balanced', random_state=random_seed, max_iter=1000, solver='liblinear')
    elif classifier_name == 'svm':
        logging.info("Using SVM meta-classifier.")
        return SVC(probability=True, random_state=random_seed, class_weight='balanced')
    elif classifier_name == 'random_forest':
        logging.info("Using Random Forest meta-classifier.")
        return RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight='balanced')
    elif classifier_name == 'lightgbm':
        logging.info("Using LightGBM meta-classifier.")
        return lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=random_seed, verbose=-1)
    else:
        raise ValueError(f"Unsupported meta-classifier type: {classifier_name}. Choose 'logistic', 'svm', 'random_forest', or 'lightgbm'.")


def train_and_predict_meta_cv(run_id_arg, run_dir_str, meta_classifier_type='logistic', n_splits_arg=5, random_seed_arg=42,
                               input_uncertainty_filename_override=None,
                               input_is_scores_filename_override=None,
                               input_generations_filename_override=None,
                               output_hybrid_meta_filename_override=None,
                               output_model_filename_override=None,
                               output_scalers_filename_override=None
                               ):
    setup_logger()
    run_dir = Path(run_dir_str)
    logging.info(f"--- Training & Predicting Hybrid Meta-Learner (K-Fold CV) for Run: {run_id_arg} using {meta_classifier_type} ---")

    unc_fname = input_uncertainty_filename_override if input_uncertainty_filename_override else DEFAULT_UNCERTAINTY_PKL
    is_scores_fname = input_is_scores_filename_override if input_is_scores_filename_override else f"{run_id_arg}{DEFAULT_IS_SCORES_CSV}"
    gen_fname = input_generations_filename_override if input_generations_filename_override else DEFAULT_GENERATIONS_PKL


    uncertainty_path = run_dir / unc_fname
    is_scores_path = run_dir / is_scores_fname
    generations_path = run_dir / gen_fname

    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)

    if not is_scores_path.is_file():
        logging.error(f"IS scores file not found: {is_scores_path}")
        return False
    df_is = pd.read_csv(is_scores_path)
    df_is['id'] = df_is['id'].astype(str).str.strip()

    if uncertainty_data is None or generations_data is None:
        logging.error(f"Uncertainty data ({uncertainty_path}), IS scores ({is_scores_path}), or Generations data ({generations_path}) missing or failed to load.")
        return False
        
    if 'internal_signal_score' not in df_is.columns:
        logging.error("IS scores CSV missing 'internal_signal_score' column.")
        return False

    response_lengths_map = {}
    task_ids_from_generations = []
    for task_id, task_details in generations_data.items():
        task_id_clean = str(task_id).strip()
        task_ids_from_generations.append(task_id_clean)
        most_likely = task_details.get("most_likely_answer", {})
        resp_len = most_likely.get("response_length")
        if resp_len is None:
            response_str = most_likely.get("response", "")
            resp_len = len(response_str) if response_str != "[GENERATION FAILED]" else 0
        response_lengths_map[task_id_clean] = resp_len
    
    df_response_lengths = pd.DataFrame(list(response_lengths_map.items()), columns=['id', 'response_length'])

    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        se_scores_raw = measures.get("semantic_entropy")
        validation_is_false_raw = uncertainty_data.get("validation_is_false")
        
        if "task_ids" in uncertainty_data and isinstance(uncertainty_data["task_ids"], list):
            task_ids_ordered = [str(tid).strip() for tid in uncertainty_data["task_ids"]]
            logging.info(f"Using task_ids order from uncertainty data ({len(task_ids_ordered)} IDs).")
        else:
            task_ids_ordered = task_ids_from_generations
            logging.info(f"Using task_ids order from generations data ({len(task_ids_ordered)} IDs), as 'task_ids' key not found in uncertainty data.")

        num_tasks = len(task_ids_ordered)

        if se_scores_raw is None or validation_is_false_raw is None:
             logging.error("SE scores or validation_is_false missing from uncertainty data.")
             return False
        
        if len(se_scores_raw) != num_tasks:
             logging.warning(f"SE scores length ({len(se_scores_raw)}) mismatch with task_ids length ({num_tasks}). Aligning.")
             se_scores_raw = (list(se_scores_raw) + [np.nan] * num_tasks)[:num_tasks]
        if len(validation_is_false_raw) != num_tasks:
             logging.warning(f"Validation_is_false length ({len(validation_is_false_raw)}) mismatch with task_ids length ({num_tasks}). Aligning.")
             validation_is_false_raw = (list(validation_is_false_raw) + [False] * num_tasks)[:num_tasks]

        df_se = pd.DataFrame({'id': task_ids_ordered, 'semantic_entropy': se_scores_raw, 'is_incorrect_auto': validation_is_false_raw})
    except Exception as e:
        logging.error(f"Error processing PKL data: {e}", exc_info=True)
        return False

    merged_df_all = pd.merge(df_se, df_is[['id', 'internal_signal_score']], on='id', how='left')
    merged_df_all = pd.merge(merged_df_all, df_response_lengths, on='id', how='left')
    
    feature_cols = ['semantic_entropy', 'internal_signal_score', 'response_length']
    merged_df = merged_df_all.dropna(subset=feature_cols + ['is_incorrect_auto']).copy()

    logging.info(f"Using {len(merged_df)} complete samples for meta-learner CV from columns: {feature_cols}.")
    if len(merged_df) < n_splits_arg or len(merged_df['is_incorrect_auto'].unique()) < 2:
        logging.error(f"Insufficient data or classes ({len(merged_df)} samples, {len(merged_df['is_incorrect_auto'].unique())} classes) for {n_splits_arg}-Fold CV.")
        return False

    X = merged_df[feature_cols].values
    y = merged_df['is_incorrect_auto'].astype(bool).values
    ids = merged_df['id'].values

    skf = StratifiedKFold(n_splits=n_splits_arg, shuffle=True, random_state=random_seed_arg)
    oof_predictions = np.full(len(y), np.nan)
    oof_ids = np.full(len(y), None, dtype=object)
    fold_aurocs = []

    logging.info(f"Starting {n_splits_arg}-Fold CV for Hybrid Meta using {meta_classifier_type}...")

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        logging.info(f"--- Processing Fold {fold_idx + 1}/{n_splits_arg} ---")
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        ids_test_fold = ids[test_index]

        if len(np.unique(y_train_fold)) < 2:
             logging.warning(f"Fold {fold_idx + 1}: Training split has only one class. Skipping fold.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test_fold
             continue

        scalers_fold = [MinMaxScaler() for _ in feature_cols]
        X_train_scaled_fold = np.zeros_like(X_train_fold, dtype=float)
        X_test_scaled_fold = np.zeros_like(X_test_fold, dtype=float)

        for i in range(X_train_fold.shape[1]):
            try:
                X_train_scaled_fold[:, i] = scalers_fold[i].fit_transform(X_train_fold[:, i].reshape(-1, 1)).flatten()
                X_test_scaled_fold[:, i] = scalers_fold[i].transform(X_test_fold[:, i].reshape(-1, 1)).flatten()
            except ValueError: 
                logging.warning(f"Fold {fold_idx + 1}: Feature {feature_cols[i]} constant in training. Setting normalized to 0.5.")
                X_train_scaled_fold[:, i] = 0.5
                X_test_scaled_fold[:, i] = 0.5
        
        try:
            classifier_fold = get_meta_classifier(meta_classifier_type, random_seed_arg)
            classifier_fold.fit(X_train_scaled_fold, y_train_fold)
            logging.info(f"Fold {fold_idx + 1}: Meta-Learner trained.")
        except Exception as e:
             logging.error(f"Fold {fold_idx + 1}: Error during meta training with {meta_classifier_type}: {e}. Skipping predictions.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test_fold
             continue

        try:
            probabilities_test = classifier_fold.predict_proba(X_test_scaled_fold)
            
            true_class_column_idx = -1
            if True in classifier_fold.classes_:
                true_class_column_idx = np.where(classifier_fold.classes_ == True)[0][0]
            elif 1 in classifier_fold.classes_:
                true_class_column_idx = np.where(classifier_fold.classes_ == 1)[0][0]


            if true_class_column_idx == -1:
                logging.error(f"Fold {fold_idx+1}: Could not find True (or 1) class index in meta classes: {classifier_fold.classes_}. Cannot get P(Incorrect).")
                oof_predictions[test_index] = np.nan
            else:
                oof_preds_fold_p_incorrect = probabilities_test[:, true_class_column_idx]
                oof_predictions[test_index] = oof_preds_fold_p_incorrect

                if len(np.unique(y_test_fold)) == 2:
                    fold_auc = roc_auc_score(y_test_fold.astype(int), oof_preds_fold_p_incorrect)
                    fold_aurocs.append(fold_auc)
                    logging.info(f"Fold {fold_idx + 1}: Test AUROC (for P(Incorrect)) = {fold_auc:.4f}")
                else:
                    logging.warning(f"Fold {fold_idx + 1}: Test split has only one class, cannot calculate fold AUROC.")
            
            oof_ids[test_index] = ids_test_fold

        except Exception as e:
            logging.error(f"Fold {fold_idx + 1}: Error during meta prediction/evaluation: {e}", exc_info=True)
            oof_predictions[test_index] = np.nan
            oof_ids[test_index] = ids_test_fold

    logging.info(f"K-Fold CV finished for Hybrid Meta ({meta_classifier_type}).")
    if fold_aurocs: logging.info(f"Average OOF AUROC (for P(Incorrect)): {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")

    if not np.all(oof_ids != None):
        missing_oof_id_mask = (oof_ids == None)
        if np.any(missing_oof_id_mask):
             logging.warning(f"Hybrid Meta: {np.sum(missing_oof_id_mask)} OOF IDs were None, filling with original IDs from merged_df.")
             oof_ids[missing_oof_id_mask] = ids[missing_oof_id_mask]

    results_oof_df = pd.DataFrame({
        'id': oof_ids,
        'hybrid_meta_score': oof_predictions
    })
    results_oof_df['id'] = results_oof_df['id'].astype(str).str.strip()

    output_hybrid_meta_fname_final = output_hybrid_meta_filename_override if output_hybrid_meta_filename_override else f"{run_id_arg}{DEFAULT_HYBRID_META_CSV}"
    output_path_all_scores = run_dir / output_hybrid_meta_fname_final
    try:
        results_oof_df.to_csv(output_path_all_scores, index=False)
        logging.info(f"Saved K-Fold OOF Hybrid Meta scores ({meta_classifier_type}, P(Incorrect)) ({len(results_oof_df)} rows, {results_oof_df['hybrid_meta_score'].isnull().sum()} NaNs) to: {output_path_all_scores}")
    except IOError as e:
        logging.error(f"Error writing OOF Hybrid Meta scores file: {e}")
        return False

    logging.info(f"Training final meta model ({meta_classifier_type}) on ALL {len(X)} valid data samples...")
    try:
        final_scalers = [MinMaxScaler() for _ in feature_cols]
        X_full_scaled = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            try:
                X_full_scaled[:, i] = final_scalers[i].fit_transform(X[:, i].reshape(-1,1)).flatten()
            except ValueError:
                logging.warning(f"Final Scaling: Feature {feature_cols[i]} constant. Setting normalized to 0.5.")
                X_full_scaled[:, i] = 0.5
        
        final_classifier = get_meta_classifier(meta_classifier_type, random_seed_arg)
        final_classifier.fit(X_full_scaled, y)

        output_model_fname_final = output_model_filename_override if output_model_filename_override else f"{run_id_arg}{DEFAULT_MODEL_PKL}"
        final_model_path = run_dir / output_model_fname_final
        with open(final_model_path, 'wb') as f_model:
            pickle.dump(final_classifier, f_model)
        logging.info(f"Saved FINAL meta model ({meta_classifier_type}) to: {final_model_path}")
        
        output_scalers_fname_final = output_scalers_filename_override if output_scalers_filename_override else f"{run_id_arg}{DEFAULT_SCALERS_PKL}"
        final_scalers_path = run_dir / output_scalers_fname_final
        with open(final_scalers_path, 'wb') as f_scaler: pickle.dump(final_scalers, f_scaler)
        logging.info(f"Saved FINAL fitted scalers to: {final_scalers_path}")

    except Exception as e:
        logging.error(f"Error training/saving final meta model/scalers ({meta_classifier_type}): {e}", exc_info=True)
        return False

    logging.info(f"--- Hybrid Meta-Learner CV Training & OOF Prediction Complete ({meta_classifier_type}): {run_id_arg} ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid meta-learner with CV & predict OOF scores.")
    parser.add_argument("run_id", type=str)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--meta_classifier", type=str, default='logistic', choices=['logistic', 'svm', 'random_forest', 'lightgbm'])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input_uncertainty_filename_override", default=None)
    parser.add_argument("--input_is_scores_filename_override", default=None)
    parser.add_argument("--input_generations_filename_override", default=None)
    parser.add_argument("--output_hybrid_meta_filename_override", default=None)
    parser.add_argument("--output_model_filename_override", default=None)
    parser.add_argument("--output_scalers_filename_override", default=None)

    args = parser.parse_args()

    train_and_predict_meta_cv(
        args.run_id, 
        args.run_dir, 
        meta_classifier_type=args.meta_classifier, 
        n_splits_arg=args.n_splits, 
        random_seed_arg=args.seed,
        input_uncertainty_filename_override=args.input_uncertainty_filename_override,
        input_is_scores_filename_override=args.input_is_scores_filename_override,
        input_generations_filename_override=args.input_generations_filename_override,
        output_hybrid_meta_filename_override=args.output_hybrid_meta_filename_override,
        output_model_filename_override=args.output_model_filename_override,
        output_scalers_filename_override=args.output_scalers_filename_override
    )