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
from sklearn.metrics import roc_auc_score
from src.utils.logging_utils import setup_logger 
from src.utils.utils import load_pickle

def prepare_probe_data(generations_data, accuracy_threshold=0.5, explicit_labels=None, task_ids_for_labels=None):    
    """
    Prepares data for internal signal probe by extracting hidden states, labels, and task IDs
    from generations data.

    Parameters
    ----------
    generations_data : dict
        A dictionary of task ID to generation details, including the
        'most_likely_answer' and its 'accuracy' and 'embedding'.
    accuracy_threshold : float, optional
        The accuracy threshold above which a generation is considered correct.
    explicit_labels : list, optional
        A list of explicit labels corresponding to the task IDs in
        task_ids_for_labels.
    task_ids_for_labels : list, optional
        A list of task IDs for which explicit labels are provided in
        explicit_labels.

    Returns
    -------
    hidden_states : list
        A list of hidden state tensors extracted from generations data.
    labels_derived : list
        A list of labels derived from generations data, either from explicit
        labels or from accuracy scores.
    task_ids_from_gen : list
        A list of task IDs from generations data.
    """
    hidden_states = []
    labels_derived = []
    task_ids_from_gen = []
    
    logging.info("Preparing data for internal signal probe...")
    processed_count = 0
    skipped_count = 0

    temp_labels_map = {}
    if explicit_labels is not None and task_ids_for_labels is not None and len(explicit_labels) == len(task_ids_for_labels):
        temp_labels_map = {str(tid).strip(): label for tid, label in zip(task_ids_for_labels, explicit_labels)}
        logging.info(f"Using {len(temp_labels_map)} explicit labels for probe data preparation.")

    for task_id_str, task_details in generations_data.items():
        task_id_clean = str(task_id_str).strip()
        try:
            most_likely = task_details.get("most_likely_answer", {})
            embedding = most_likely.get("embedding")
            
            current_label_is_correct = None
            if task_id_clean in temp_labels_map:
                current_label_is_correct = temp_labels_map[task_id_clean]
            else:
                accuracy = most_likely.get("accuracy")
                if accuracy is not None:
                    current_label_is_correct = accuracy > accuracy_threshold
                else:
                    logging.warning(f"Task {task_id_clean} missing explicit label and accuracy field. Skipping.")
                    skipped_count += 1
                    continue
            
            if embedding is not None and current_label_is_correct is not None:
                try:
                    if isinstance(embedding, torch.Tensor):
                        hs_tensor = embedding.cpu()
                    else:
                        hs_tensor = torch.tensor(embedding).cpu()
                    hidden_states.append(hs_tensor)
                    labels_derived.append(current_label_is_correct)
                    task_ids_from_gen.append(task_id_clean)
                    processed_count += 1
                except Exception as conversion_e:
                    logging.warning(f"Could not process embedding for {task_id_clean}: {conversion_e}. Skipping.")
                    skipped_count += 1
                    continue
            else:
                if embedding is None: logging.warning(f"Task {task_id_clean} missing embedding. Skipping.")
                skipped_count += 1
        except Exception as e:
            logging.error(f"Error processing task {task_id_clean}: {e}. Skipping.")
            skipped_count += 1
            
    logging.info(f"Probe data preparation complete. Processed: {processed_count}, Skipped: {skipped_count}")
    if not hidden_states:
        logging.error("No valid hidden states extracted for probe.")
        return None, None, None
    return hidden_states, labels_derived, task_ids_from_gen


def get_classifier(classifier_name, random_seed):
    """
    Return a classifier instance given its name and random seed.

    Parameters
    ----------
    classifier_name : str
        The name of the classifier. Choose 'logistic', 'svm', or 'random_forest'.
    random_seed : int
        The random seed for the classifier.

    Returns
    -------
    A trained classifier instance.
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


def run_internal_signal_probe_cv(args_run_id, args_base_dir, args_classifier_type, args_n_splits, args_random_seed, args_probe_accuracy_threshold,
                                 input_generations_filename_override=None,
                                 input_uncertainty_filename_override=None,
                                 output_scores_filename_override=None,
                                 output_model_filename_override=None):
    """
    Run internal signal probe using K-Fold cross-validation and predict on all samples.

    Parameters
    ----------
    args_run_id : str
        Run ID.
    args_base_dir : str
        Dir containing generations PKL and potentially uncertainty PKL.
    args_classifier_type : str
        Classifier type to use. Choose 'logistic', 'svm', or 'random_forest'.
    args_n_splits : int
        Number of folds for K-Fold CV.
    args_random_seed : int
        Random seed for classifier and K-Fold CV.
    args_probe_accuracy_threshold : float
        Accuracy threshold above which a generation is considered correct.
    input_generations_filename_override : str, optional
        Override name for input generations PKL file.
    input_uncertainty_filename_override : str, optional
        Override name for input uncertainty PKL file.
    output_scores_filename_override : str, optional
        Override name for output K-Fold OOF scores CSV file.
    output_model_filename_override : str, optional
        Override name for output final probe model PKL file.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    setup_logger()
    run_dir = Path(args_base_dir)
    logging.info(f"--- Running IS Probe K-Fold CV & Full Prediction for Run: {args_run_id} using {args_classifier_type} ---")
    
    gen_fname = input_generations_filename_override if input_generations_filename_override else "validation_generations.pkl"
    generations_path = run_dir / gen_fname
    generations_data = load_pickle(generations_path)
    if generations_data is None: 
        logging.error(f"IS Probe: Generations data not found at {generations_path}"); return False

    explicit_labels_for_probe = None
    task_ids_for_explicit_labels = None

    if input_uncertainty_filename_override:
        unc_fname = input_uncertainty_filename_override
        uncertainty_path = run_dir / unc_fname
        uncertainty_data = load_pickle(uncertainty_path)
        if uncertainty_data and "validation_is_false" in uncertainty_data and "task_ids" in uncertainty_data:
            validation_is_false_list = uncertainty_data["validation_is_false"]
            explicit_labels_for_probe = [not is_false_item for is_false_item in validation_is_false_list]
            task_ids_for_explicit_labels = uncertainty_data["task_ids"]
            logging.info(f"IS Probe: Loaded explicit labels from {unc_fname}")
        elif uncertainty_data and "validation_is_false" in uncertainty_data:
             validation_is_false_list = uncertainty_data["validation_is_false"]
             explicit_labels_for_probe = [not is_false_item for is_false_item in validation_is_false_list]
             task_ids_for_explicit_labels = list(generations_data.keys())
             if len(explicit_labels_for_probe) != len(task_ids_for_explicit_labels):
                 logging.warning(f"Length mismatch between explicit labels ({len(explicit_labels_for_probe)}) and task IDs from generations ({len(task_ids_for_explicit_labels)}). Probe may be unreliable.")
                 min_len = min(len(explicit_labels_for_probe), len(task_ids_for_explicit_labels))
                 explicit_labels_for_probe = explicit_labels_for_probe[:min_len]
                 task_ids_for_explicit_labels = task_ids_for_explicit_labels[:min_len]
             logging.info(f"IS Probe: Loaded explicit labels from {unc_fname}, assuming order alignment with generations data.")
        else:
            logging.warning(f"IS Probe: Could not load 'validation_is_false' from {uncertainty_path} or 'task_ids' missing. Will derive labels from generations 'accuracy'.")
            explicit_labels_for_probe = None
            task_ids_for_explicit_labels = None
    
    hidden_states, labels, task_ids_all = prepare_probe_data(generations_data, 
                                                             accuracy_threshold=args_probe_accuracy_threshold,
                                                             explicit_labels=explicit_labels_for_probe,
                                                             task_ids_for_labels=task_ids_for_explicit_labels)

    if hidden_states is None or labels is None: 
        logging.error("IS Probe: Failed to prepare probe data.")
        return False
    if len(np.unique(labels)) < 2: 
        logging.error("IS Probe: Only one class label found. Cannot train/evaluate probe.")
        return False

    logging.info(f"IS Probe: Prepared {len(hidden_states)} embeddings and {len(labels)} labels for K-Fold CV.")

    try:
        X_full = np.array([vec.numpy().flatten() for vec in hidden_states])
        y_full = np.array(labels)
        ids_full = np.array(task_ids_all)
    except Exception as e:
        logging.error(f"IS Probe: Error converting embeddings/labels to NumPy array: {e}"); return False

    skf = StratifiedKFold(n_splits=args_n_splits, shuffle=True, random_state=args_random_seed)
    oof_predictions_proba_false = np.full(len(y_full), np.nan)
    oof_ids = np.full(len(y_full), None, dtype=object)

    logging.info(f"IS Probe: Starting {args_n_splits}-Fold CV using {args_classifier_type}...")
    fold_aurocs = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_full, y_full)):
        logging.info(f"--- IS Probe: Processing Fold {fold_idx + 1}/{args_n_splits} ---")
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train_fold, y_test_fold = y_full[train_index], y_full[test_index]
        ids_test_fold = ids_full[test_index]

        if len(np.unique(y_train_fold)) < 2:
             logging.warning(f"IS Probe Fold {fold_idx + 1}: Training split has only one class. Skipping fold.")
             oof_predictions_proba_false[test_index] = np.nan
             oof_ids[test_index] = ids_test_fold
             continue

        try:
            classifier_fold = get_classifier(args_classifier_type, args_random_seed)
            classifier_fold.fit(X_train, y_train_fold)
            logging.info(f"IS Probe Fold {fold_idx + 1}: Probe trained.")
        except Exception as e:
             logging.error(f"IS Probe Fold {fold_idx + 1}: Error during training: {e}. Skipping.")
             oof_predictions_proba_false[test_index] = np.nan
             oof_ids[test_index] = ids_test_fold
             continue

        try:
            probabilities_test = classifier_fold.predict_proba(X_test)
            
            false_class_column_idx = -1
            if False in classifier_fold.classes_:
                false_class_column_idx = np.where(classifier_fold.classes_ == False)[0][0]
            elif 0 in classifier_fold.classes_:
                false_class_column_idx = np.where(classifier_fold.classes_ == 0)[0][0]

            if false_class_column_idx == -1 :
                 logging.error(f"IS Probe Fold {fold_idx+1}: Could not find False (or 0) class index in classifier classes: {classifier_fold.classes_}. Cannot get P(Incorrect).")
                 oof_predictions_proba_false[test_index] = np.nan
            else:
                oof_preds_fold_p_false = probabilities_test[:, false_class_column_idx]
                oof_predictions_proba_false[test_index] = oof_preds_fold_p_false

                if len(np.unique(y_test_fold)) == 2:
                     fold_auc = roc_auc_score((y_test_fold == False).astype(int), oof_preds_fold_p_false)
                     fold_aurocs.append(fold_auc)
                     logging.info(f"IS Probe Fold {fold_idx + 1}: Test AUROC (for P(Incorrect)) = {fold_auc:.4f}")
                else:
                     logging.warning(f"IS Probe Fold {fold_idx + 1}: Test split has only one class, cannot calculate fold AUROC.")
            
            oof_ids[test_index] = ids_test_fold

        except Exception as e:
            logging.error(f"IS Probe Fold {fold_idx + 1}: Error during prediction/evaluation: {e}")
            oof_predictions_proba_false[test_index] = np.nan
            oof_ids[test_index] = ids_test_fold

    logging.info(f"IS Probe K-Fold CV finished for {args_classifier_type}.")
    if fold_aurocs: logging.info(f"IS Probe Average Out-of-Fold AUROC (for P(Incorrect)): {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")

    if not np.all(oof_ids != None):
        missing_indices = np.where(oof_ids == None)[0]
        logging.warning(f"IS Probe: {len(missing_indices)} OOF IDs were None, filling with original IDs.")
        if len(missing_indices) > 0 : oof_ids[missing_indices] = ids_full[missing_indices]


    results_oof_df = pd.DataFrame({
        'id': oof_ids,
        'internal_signal_score': oof_predictions_proba_false
    })
    results_oof_df['id'] = results_oof_df['id'].astype(str).str.strip()

    output_scores_fname_final = output_scores_filename_override if output_scores_filename_override else f"{args_run_id}_internal_signal_scores_all.csv"
    output_path_all_scores = run_dir / output_scores_fname_final
    try:
        results_oof_df.to_csv(output_path_all_scores, index=False)
        logging.info(f"IS Probe: Saved K-Fold OOF IS scores (P(Incorrect)) ({len(results_oof_df)} rows, {results_oof_df['internal_signal_score'].isnull().sum()} NaNs) to: {output_path_all_scores}")
    except IOError as e: logging.error(f"IS Probe: Error writing OOF IS scores file: {e}"); return False

    logging.info(f"IS Probe: Training final probe model ({args_classifier_type}) on ALL {len(X_full)} valid data samples...")
    try:
        final_classifier = get_classifier(args_classifier_type, args_random_seed)
        final_classifier.fit(X_full, y_full)
        
        output_model_fname_final = output_model_filename_override if output_model_filename_override else f"{args_run_id}_probe_model.pkl"
        final_probe_model_path = run_dir / output_model_fname_final
        with open(final_probe_model_path, 'wb') as f_probe: pickle.dump(final_classifier, f_probe)
        logging.info(f"IS Probe: Saved final probe model trained on all data to: {final_probe_model_path}")
    except Exception as e:
        logging.error(f"IS Probe: Error training/saving final probe model: {e}", exc_info=True)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IS probe using K-Fold CV and predict on all samples.")
    parser.add_argument("run_id", type=str, help="Run ID.")
    parser.add_argument("--base_dir", type=str, required=True, help="Dir containing generations PKL and potentially uncertainty PKL.")
    parser.add_argument("--classifier", type=str, default='logistic', choices=['logistic', 'svm', 'random_forest'])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe_accuracy_threshold", type=float, default=0.5)

    parser.add_argument("--input_generations_filename_override", default=None)
    parser.add_argument("--input_uncertainty_filename_override", default=None)
    parser.add_argument("--output_scores_filename_override", default=None)
    parser.add_argument("--output_model_filename_override", default=None)
    
    args = parser.parse_args()

    run_internal_signal_probe_cv(
        args.run_id,
        args.base_dir,
        args.classifier,
        args.n_splits,
        args.seed,
        args.probe_accuracy_threshold,
        args.input_generations_filename_override,
        args.input_uncertainty_filename_override,
        args.output_scores_filename_override,
        args.output_model_filename_override
    )