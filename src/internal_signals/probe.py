import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle

def prepare_probe_data(generations_data, accuracy_threshold=0.5):
    """
    Prepare the data for internal signal probing.

    Parameters
    ----------
    generations_data : dict
        The generations data loaded from a pickle file.
    accuracy_threshold : float, default=0.5
        The accuracy threshold above which a generation is considered correct.

    Returns
    -------
    hidden_states : list of torch.Tensors
        The internal hidden state embeddings.
    labels : list of bool
        The labels (True if correct, False otherwise) for each task.
    task_ids : list of str
        The task IDs corresponding to the above data.
    """
    hidden_states = []; labels = []; task_ids = []
    logging.info("Preparing data for internal signal probe...")
    processed_count = 0; skipped_count = 0
    for task_id, task_details in generations_data.items():
        try:
            most_likely = task_details.get("most_likely_answer", {})
            embedding = most_likely.get("embedding"); accuracy = most_likely.get("accuracy")
            if embedding is not None and accuracy is not None:
                try:
                    if isinstance(embedding, torch.Tensor): hs_tensor = embedding.cpu()
                    else: hs_tensor = torch.tensor(embedding).cpu()
                    hidden_states.append(hs_tensor)
                    is_correct = accuracy > accuracy_threshold; labels.append(is_correct); task_ids.append(task_id); processed_count += 1
                except Exception as conversion_e: logging.warning(f"Could not process embedding for {task_id}: {conversion_e}. Skipping."); skipped_count += 1; continue
            else: logging.warning(f"Task {task_id} missing embedding/accuracy. Skipping."); skipped_count += 1
        except Exception as e: logging.error(f"Error processing task {task_id}: {e}. Skipping."); skipped_count += 1
    logging.info(f"Data preparation complete. Processed: {processed_count}, Skipped: {skipped_count}")
    if not hidden_states: logging.error("No valid hidden states extracted."); return None, None, None
    return hidden_states, labels, task_ids


def run_internal_signal_probe_cv(run_id, base_dir, n_splits=5, random_seed=42):
    """
    Runs the internal signal probe using K-Fold Cross-Validation.

    The internal signal probe is a binary logistic regression model that is trained to predict whether a generation is correct or not based on the internal state embeddings of the model. The probe is evaluated using K-Fold Cross-Validation, and the out-of-fold predictions are saved to a CSV file.

    Parameters
    ----------
    run_id : str
        The ID of the run.
    base_dir : str
        The base directory where the run data is located.
    n_splits : int, default=5
        The number of folds for K-Fold Cross-Validation.
    random_seed : int, default=42
        The random seed for the logistic regression model.

    Returns
    -------
    bool
        True if the probe was successfully trained and evaluated, False otherwise.
    """
    setup_logger()
    run_dir = Path(base_dir)
    logging.info(f"--- Running IS Probe K-Fold CV & Full Prediction for Run: {run_id} ---")

    generations_path = run_dir / "validation_generations.pkl"
    generations_data = load_pickle(generations_path)
    if generations_data is None: return False

    hidden_states, labels, task_ids_all = prepare_probe_data(generations_data)
    if hidden_states is None: return False
    if len(np.unique(labels)) < 2: logging.error("Only one class label found. Cannot train/evaluate probe."); return False

    logging.info(f"Prepared {len(hidden_states)} embeddings and labels for K-Fold CV.")

    try:
        X_full = np.array([vec.numpy().flatten() for vec in hidden_states])
        y_full = np.array(labels)
        ids_full = np.array(task_ids_all)
    except Exception as e:
        logging.error(f"Error converting embeddings to NumPy array: {e}"); return False

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_predictions = np.full(len(y_full), np.nan)
    oof_ids = np.full(len(y_full), None, dtype=object)

    logging.info(f"Starting {n_splits}-Fold Cross-Validation for IS probe...")
    fold_aurocs = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_full, y_full)):
        logging.info(f"--- Processing Fold {fold_idx + 1}/{n_splits} ---")
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]
        ids_test = ids_full[test_index]

        if len(np.unique(y_train)) < 2:
             logging.warning(f"Fold {fold_idx + 1}: Training split has only one class. Probe might not be effective. Skipping fold evaluation.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test
             continue
        
        try:
            classifier_fold = LogisticRegression(random_state=random_seed, class_weight="balanced", max_iter=1000)
            classifier_fold.fit(X_train, y_train)
            logging.info(f"Fold {fold_idx + 1}: Probe trained.")
        except Exception as e:
             logging.error(f"Fold {fold_idx + 1}: Error during training: {e}. Skipping predictions for this fold.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test
             continue

        try:
            probabilities_test = classifier_fold.predict_proba(X_test)
            false_class_idx = np.where(classifier_fold.classes_ == False)[0][0]
            oof_preds_fold = probabilities_test[:, false_class_idx]

            oof_predictions[test_index] = oof_preds_fold
            oof_ids[test_index] = ids_test

            if len(np.unique(y_test)) == 2:
                fold_auc = roc_auc_score(y_test == False, oof_preds_fold)
                fold_aurocs.append(fold_auc)
                logging.info(f"Fold {fold_idx + 1}: Test AUROC = {fold_auc:.4f}")
            else:
                 logging.warning(f"Fold {fold_idx + 1}: Test split has only one class, cannot calculate fold AUROC.")

        except Exception as e:
            logging.error(f"Fold {fold_idx + 1}: Error during prediction/evaluation: {e}")
            oof_predictions[test_index] = np.nan
            oof_ids[test_index] = ids_test

    logging.info("K-Fold Cross-Validation finished.")
    if fold_aurocs: logging.info(f"Average Out-of-Fold AUROC: {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")

    if not np.all(oof_ids != None):
        missing_ids_indices = np.where(oof_ids == None)[0]
        logging.error(f"CRITICAL ERROR: {len(missing_ids_indices)} IDs were not assigned during K-Fold. Check logic.")
        for idx in missing_ids_indices:
             oof_ids[idx] = ids_full[idx]


    results_oof_df = pd.DataFrame({
        'id': oof_ids,
        'internal_signal_score': oof_predictions
    })
    results_oof_df['id'] = results_oof_df['id'].astype(str).str.strip()

    output_path_all = run_dir / f"{run_id}_internal_signal_scores_all.csv"
    try:
        results_oof_df.to_csv(output_path_all, index=False)
        logging.info(f"Saved K-Fold Out-of-Fold IS scores ({len(results_oof_df)} rows, {results_oof_df['internal_signal_score'].isnull().sum()} NaNs) to: {output_path_all}")
    except IOError as e: logging.error(f"Error writing OOF IS scores file: {e}"); return False

    logging.info("Training final probe model on ALL valid data...")
    try:
        final_classifier = LogisticRegression(random_state=random_seed, class_weight="balanced", max_iter=1000)
        final_classifier.fit(X_full, y_full)
        probe_model_path = run_dir / f"{run_id}_probe_model.pkl"
        with open(probe_model_path, 'wb') as f_probe: pickle.dump(final_classifier, f_probe)
        logging.info(f"Saved final probe model trained on all data to: {probe_model_path}")
    except Exception as e:
        logging.error(f"Error training/saving final probe model: {e}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IS probe using K-Fold CV and predict on all samples.")
    parser.add_argument("run_id", type=str, help="Run ID.")
    parser.add_argument("--base_dir", type=str, required=True, help="Dir containing validation_generations.pkl.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold CV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_internal_signal_probe_cv(args.run_id, args.base_dir, args.n_splits, args.seed)