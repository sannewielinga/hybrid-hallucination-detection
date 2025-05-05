import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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


def get_classifier(classifier_name, random_seed):
    """
    Return a classifier object given its name and random seed.

    Parameters
    ----------
    classifier_name : str
        One of 'logistic', 'svm', or 'random_forest'.
    random_seed : int
        The random seed for the classifier.

    Returns
    -------
    classifier : object
        The classifier object.
    """
    if classifier_name == 'logistic':
        logging.info("Using Logistic Regression classifier.")
        return LogisticRegression(random_state=random_seed, class_weight="balanced", max_iter=2000, solver='liblinear')
    elif classifier_name == 'svm':
        logging.info("Using SVM classifier.")
        return SVC(probability=True, random_state=random_seed, class_weight='balanced')
    elif classifier_name == 'random_forest':
        logging.info("Using Random Forest classifier.")
        return RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight='balanced')
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_name}. Choose 'logistic', 'svm', or 'random_forest'.")


def run_internal_signal_probe_cv(run_id, base_dir, classifier_type='logistic', n_splits=5, random_seed=42, probe_accuracy_threshold=0.5):
    """
    Run internal signal probe cross-validation and prediction.

    This function prepares data, performs K-Fold cross-validation, evaluates
    the performance of the internal signal probe using a specified classifier,
    and saves the results. It also trains a final probe model on all available
    data.

    Parameters
    ----------
    run_id : str
        The identifier for the current run.
    base_dir : str
        The base directory containing the required data files.
    classifier_type : str, optional
        The type of classifier to use ('logistic', 'svm', or 'random_forest').
        Defaults to 'logistic'.
    n_splits : int, optional
        Number of folds for K-Fold cross-validation. Defaults to 5.
    random_seed : int, optional
        The random seed for reproducibility. Defaults to 42.
    probe_accuracy_threshold : float, optional
        The threshold for defining probe correctness. Defaults to 0.5.

    Returns
    -------
    bool
        True if the process completes successfully, False otherwise.
    """

    setup_logger()
    run_dir = Path(base_dir)
    logging.info(f"--- Running IS Probe K-Fold CV & Full Prediction for Run: {run_id} using {classifier_type} ---")
    generations_path = run_dir / "validation_generations.pkl"
    generations_data = load_pickle(generations_path)
    if generations_data is None: return False

    hidden_states, labels, task_ids_all = prepare_probe_data(generations_data, accuracy_threshold=probe_accuracy_threshold)

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

    logging.info(f"Starting {n_splits}-Fold Cross-Validation for IS probe using {classifier_type}...")
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
            classifier_fold = get_classifier(classifier_type, random_seed)
            classifier_fold.fit(X_train, y_train)
            logging.info(f"Fold {fold_idx + 1}: Probe trained.")
        except Exception as e:
             logging.error(f"Fold {fold_idx + 1}: Error during training with {classifier_type}: {e}. Skipping predictions for this fold.")
             oof_predictions[test_index] = np.nan
             oof_ids[test_index] = ids_test
             continue

        try:
            probabilities_test = classifier_fold.predict_proba(X_test)
            false_class_idx_list = np.where(classifier_fold.classes_ == False)[0]
            if not len(false_class_idx_list):
                 logging.error(f"Fold {fold_idx+1}: Could not find False class index in classifier classes: {classifier_fold.classes_}. Skipping fold eval.")
                 oof_predictions[test_index] = np.nan
                 oof_ids[test_index] = ids_test
                 continue
            false_class_idx = false_class_idx_list[0]

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
            logging.error(f"Fold {fold_idx + 1}: Error during prediction/evaluation with {classifier_type}: {e}")
            oof_predictions[test_index] = np.nan
            oof_ids[test_index] = ids_test

    logging.info(f"K-Fold Cross-Validation finished for {classifier_type}.")
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
        logging.info(f"Saved K-Fold Out-of-Fold IS ({classifier_type}) scores ({len(results_oof_df)} rows, {results_oof_df['internal_signal_score'].isnull().sum()} NaNs) to: {output_path_all}")
    except IOError as e: logging.error(f"Error writing OOF IS scores file: {e}"); return False

    logging.info(f"Training final probe model ({classifier_type}) on ALL valid data...")
    try:
        final_classifier = get_classifier(classifier_type, random_seed)
        final_classifier.fit(X_full, y_full)
        probe_model_path = run_dir / f"{run_id}_probe_model.pkl"
        with open(probe_model_path, 'wb') as f_probe: pickle.dump(final_classifier, f_probe)
        logging.info(f"Saved final probe model ({classifier_type}) trained on all data to: {probe_model_path}")
    except Exception as e:
        logging.error(f"Error training/saving final probe model ({classifier_type}): {e}")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IS probe using K-Fold CV and predict on all samples.")
    parser.add_argument("run_id", type=str, help="Run ID.")
    parser.add_argument("--base_dir", type=str, required=True, help="Dir containing validation_generations.pkl.")
    parser.add_argument("--classifier", type=str, default='logistic', choices=['logistic', 'svm', 'random_forest'], help="Classifier type for the IS probe.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold CV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--probe_accuracy_threshold", type=float, default=0.5, help="Accuracy threshold for defining correctness.")

    args = parser.parse_args()

    run_internal_signal_probe_cv(
        args.run_id,
        args.base_dir,
        classifier_type=args.classifier,
        n_splits=args.n_splits,
        random_seed=args.seed,
        probe_accuracy_threshold=args.probe_accuracy_threshold
    )