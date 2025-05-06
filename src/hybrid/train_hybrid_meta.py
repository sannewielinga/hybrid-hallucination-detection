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

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle

def get_meta_classifier(classifier_name, random_seed):
    """
    Return a meta-classifier object given its name and random seed.

    Parameters
    ----------
    classifier_name : str
        One of 'logistic', 'svm', 'random_forest', or 'lightgbm'.
    random_seed : int
        The random seed for the classifier.

    Returns
    -------
    classifier : object
        The classifier object.
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
        return lgb.LGBMClassifier(objective='binary', class_weight='balanced', random_state=random_seed)
    else:
        raise ValueError(f"Unsupported meta-classifier type: {classifier_name}. Choose 'logistic', 'svm', 'random_forest', or 'lightgbm'.")


def train_and_predict_meta_cv(run_id, run_dir, meta_classifier_type='logistic', n_splits=5, random_seed=42):
    """
    Train and predict a hybrid meta-learner using K-Fold Cross-Validation.

    Parameters
    ----------
    run_id : str
        The ID of the run.
    run_dir : str
        The directory of the run.
    meta_classifier_type : str, optional
        The type of meta-classifier to use (default is 'logistic'). One of 'logistic', 'svm', 'random_forest', or 'lightgbm'.
    n_splits : int, optional
        The number of folds for K-Fold CV (default is 5).
    random_seed : int, optional
        The random seed for the meta-classifier (default is 42).

    Returns
    -------
    bool
        Whether the training and prediction were successful.
    """
    setup_logger()
    run_dir = Path(run_dir)
    logging.info(f"--- Training & Predicting Hybrid Meta-Learner (K-Fold CV) for Run: {run_id} using {meta_classifier_type} ---")

    uncertainty_path = run_dir / "uncertainty_measures.pkl"
    generations_path = run_dir / "validation_generations.pkl"
    is_scores_path = run_dir / f"{run_id}_internal_signal_scores_all.csv"

    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)
    if not is_scores_path.is_file():
        logging.error(f"IS scores file not found: {is_scores_path}"); return False
    df_is = pd.read_csv(is_scores_path); df_is['id'] = df_is['id'].astype(str).str.strip()

    if uncertainty_data is None or generations_data is None:
        return False
    if 'internal_signal_score' not in df_is.columns:
        logging.error("IS scores CSV missing column."); return False

    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        se_scores = measures.get("semantic_entropy")
        validation_is_false = uncertainty_data.get("validation_is_false")
        task_ids_ordered = list(generations_data.keys()); num_tasks = len(task_ids_ordered)

        if se_scores is None or validation_is_false is None:
             logging.error("SE scores or validation_is_false missing from uncertainty data.")
             return False
        if len(se_scores) != num_tasks or len(validation_is_false) != num_tasks:
             logging.warning(f"Data length mismatch: SE={len(se_scores)}, ValFalse={len(validation_is_false)}, Tasks={num_tasks}. Attempting to align based on task order.")
             min_len = min(len(se_scores), len(validation_is_false), num_tasks)
             task_ids_ordered = task_ids_ordered[:min_len]
             se_scores = se_scores[:min_len]
             validation_is_false = validation_is_false[:min_len]

        df_se = pd.DataFrame({'id': [str(tid).strip() for tid in task_ids_ordered], 'semantic_entropy': se_scores, 'is_incorrect_auto': validation_is_false})
    except Exception as e:
        logging.error(f"Error processing PKL data: {e}", exc_info=True); return False

    merged_df_all = pd.merge(df_se, df_is[['id', 'internal_signal_score']], on='id', how='inner')
    merged_df = merged_df_all.dropna(subset=['semantic_entropy', 'internal_signal_score', 'is_incorrect_auto']).copy()

    logging.info(f"Using {len(merged_df)} complete samples for meta-learner CV.")
    if len(merged_df) < n_splits or len(merged_df['is_incorrect_auto'].unique()) < 2:
        logging.error(f"Insufficient data or classes ({len(merged_df)} samples, {len(merged_df['is_incorrect_auto'].unique())} classes) for {n_splits}-Fold CV.")
        return False

    X = merged_df[['semantic_entropy', 'internal_signal_score']].values
    y = merged_df['is_incorrect_auto'].astype(bool).values
    ids = merged_df['id'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_predictions = np.full(len(y), np.nan)
    oof_ids = np.full(len(y), None, dtype=object)
    fold_aurocs = []

    logging.info(f"Starting {n_splits}-Fold Cross-Validation for Hybrid Meta using {meta_classifier_type}...")

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        logging.info(f"--- Processing Fold {fold_idx + 1}/{n_splits} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ids_test = ids[test_index]

        if len(np.unique(y_train)) < 2:
             logging.warning(f"Fold {fold_idx + 1}: Training split has only one class. Skipping fold.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test
             continue

        scaler_se_fold = MinMaxScaler()
        scaler_is_fold = MinMaxScaler()

        try:
            X_train_se = scaler_se_fold.fit_transform(X_train[:, 0].reshape(-1, 1))
            X_test_se = scaler_se_fold.transform(X_test[:, 0].reshape(-1, 1))
        except ValueError:
            logging.warning(f"Fold {fold_idx + 1}: SE constant in training fold. Setting normalized values to 0.5.")
            X_train_se = np.full_like(X_train[:, 0].reshape(-1, 1), 0.5)
            X_test_se = np.full_like(X_test[:, 0].reshape(-1, 1), 0.5)

        try:
            X_train_is = scaler_is_fold.fit_transform(X_train[:, 1].reshape(-1, 1))
            X_test_is = scaler_is_fold.transform(X_test[:, 1].reshape(-1, 1))
        except ValueError:
            logging.warning(f"Fold {fold_idx + 1}: IS constant in training fold. Setting normalized values to 0.5.")
            X_train_is = np.full_like(X_train[:, 1].reshape(-1, 1), 0.5)
            X_test_is = np.full_like(X_test[:, 1].reshape(-1, 1), 0.5)

        X_train_scaled = np.hstack((X_train_se, X_train_is))
        X_test_scaled = np.hstack((X_test_se, X_test_is))

        try:
            classifier_fold = get_meta_classifier(meta_classifier_type, random_seed)
            classifier_fold.fit(X_train_scaled, y_train)
            logging.info(f"Fold {fold_idx + 1}: Meta-Learner trained.")
        except Exception as e:
             logging.error(f"Fold {fold_idx + 1}: Error during meta training with {meta_classifier_type}: {e}. Skipping predictions.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test
             continue

        try:
            probabilities_test = classifier_fold.predict_proba(X_test_scaled)
            true_class_idx_list = np.where(classifier_fold.classes_ == True)[0]
            if not len(true_class_idx_list):
                logging.error(f"Fold {fold_idx+1}: Could not find True class index in meta classes: {classifier_fold.classes_}. Skipping.")
                oof_predictions[test_index] = np.nan
            else:
                true_class_idx = true_class_idx_list[0]
                oof_preds_fold = probabilities_test[:, true_class_idx]
                oof_predictions[test_index] = oof_preds_fold

                if len(np.unique(y_test)) == 2:
                    fold_auc = roc_auc_score(y_test, oof_preds_fold)
                    fold_aurocs.append(fold_auc)
                    logging.info(f"Fold {fold_idx + 1}: Test AUROC = {fold_auc:.4f}")
                else:
                    logging.warning(f"Fold {fold_idx + 1}: Test split has only one class, cannot calculate fold AUROC.")

            oof_ids[test_index] = ids_test

        except Exception as e:
            logging.error(f"Fold {fold_idx + 1}: Error during meta prediction/evaluation: {e}", exc_info=True)
            oof_predictions[test_index] = np.nan
            oof_ids[test_index] = ids_test

    logging.info(f"K-Fold Cross-Validation finished for Hybrid Meta ({meta_classifier_type}).")
    if fold_aurocs: logging.info(f"Average Out-of-Fold AUROC: {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")

    if not np.all(oof_ids != None):
        logging.error("Critical error: Some OOF IDs were not assigned.")
        return False

    results_oof_df = pd.DataFrame({
        'id': oof_ids,
        'hybrid_meta_score': oof_predictions
    })
    results_oof_df['id'] = results_oof_df['id'].astype(str).str.strip()

    output_path_all = run_dir / f"{run_id}_hybrid_meta_scores_all.csv"
    try:
        results_oof_df.to_csv(output_path_all, index=False)
        logging.info(f"Saved K-Fold OOF Hybrid Meta ({meta_classifier_type}) scores ({len(results_oof_df)} rows, {results_oof_df['hybrid_meta_score'].isnull().sum()} NaNs) to: {output_path_all}")
    except IOError as e: logging.error(f"Error writing OOF Hybrid Meta scores file: {e}"); return False

    logging.info(f"Training final meta model ({meta_classifier_type}) on ALL valid data...")
    try:
        final_scaler_se = MinMaxScaler()
        final_scaler_is = MinMaxScaler()
        X_full_se = X[:, 0].reshape(-1, 1)
        X_full_is = X[:, 1].reshape(-1, 1)
        try: X_full_se_norm = final_scaler_se.fit_transform(X_full_se)
        except ValueError: X_full_se_norm = np.full_like(X_full_se, 0.5)
        try: X_full_is_norm = final_scaler_is.fit_transform(X_full_is)
        except ValueError: X_full_is_norm = np.full_like(X_full_is, 0.5)

        X_full_scaled = np.hstack((X_full_se_norm, X_full_is_norm))

        final_classifier = get_meta_classifier(meta_classifier_type, random_seed)
        final_classifier.fit(X_full_scaled, y)

        final_scalers_path = run_dir / f"{run_id}_hybrid_meta_scalers.pkl"
        with open(final_scalers_path, 'wb') as f_scaler: pickle.dump({'scaler_se': final_scaler_se, 'scaler_is': final_scaler_is}, f_scaler)
        logging.info(f"Saved FINAL fitted scalers to: {final_scalers_path}")

        final_model_path = run_dir / f"{run_id}_hybrid_meta_model.pkl"
        with open(final_model_path, 'wb') as f_probe: pickle.dump(final_classifier, f_probe)
        logging.info(f"Saved FINAL meta model ({meta_classifier_type}) trained on all data to: {final_model_path}")
    except Exception as e:
        logging.error(f"Error training/saving final meta model/scalers ({meta_classifier_type}): {e}", exc_info=True)
        return False

    logging.info(f"--- Hybrid Meta-Learner CV Training & OOF Prediction Complete ({meta_classifier_type}): {run_id} ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid meta-learner with CV & predict OOF scores.")
    parser.add_argument("run_id", type=str)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--meta_classifier", type=str, default='logistic', choices=['logistic', 'svm', 'random_forest', 'lightgbm'], help="Classifier type for the meta-learner.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold CV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CV split and classifier.")

    args = parser.parse_args()

    train_and_predict_meta_cv(args.run_id, args.run_dir, meta_classifier_type=args.meta_classifier, n_splits=args.n_splits, random_seed=args.seed)