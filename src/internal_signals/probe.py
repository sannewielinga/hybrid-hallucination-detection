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

def prepare_probe_data(generations_data, metric_threshold):
    """
    Prepare the data for training the internal signal probe. This function takes in the full generations data object
    and a threshold value for the accuracy metric. It will extract the embeddings for each layer specified in the
    probe config from the generations data and concatenate them into a single tensor. The label for each sample will
    be set to 1 if the accuracy metric score is above the given threshold, and 0 otherwise.

    Args:
        generations_data (dict): Full generations data object, with task IDs as keys and task details as values.
        metric_threshold (float): Threshold value for the accuracy metric. Samples with scores above this threshold
            will be labeled as 1, and samples with scores below will be labeled as 0.

    Returns:
        tuple: A tuple containing the list of hidden states tensors, the list of labels, and the list of task IDs.
            These can be used to train the internal signal probe model.
    """
    hidden_states_list = []
    labels = []
    task_ids = []
    logging.info("Preparing data for internal signal probe (multi-layer aware)...")
    processed_count = 0
    skipped_count = 0

    sorted_layer_keys_to_use = None
    first_valid_embedding_dict = None

    for task_id_inspect in generations_data:
        candidate_embedding_obj = generations_data[task_id_inspect].get("most_likely_answer", {}).get("embedding")
        if isinstance(candidate_embedding_obj, dict) and candidate_embedding_obj:
            valid_candidate = True
            for k, v_emb in candidate_embedding_obj.items():
                if not k.startswith("layer_") or v_emb is None:
                    pass
            if any(v is not None for v in candidate_embedding_obj.values()):
                 first_valid_embedding_dict = candidate_embedding_obj
                 break

    if first_valid_embedding_dict:
        potential_keys = [k for k,v in first_valid_embedding_dict.items() if k.startswith("layer_") and v is not None]
        try:
            sorted_layer_keys_to_use = sorted(potential_keys, key=lambda x: int(x.split('_')[1]))
        except (ValueError, IndexError):
            logging.warning(f"Could not parse layer indices numerically from keys: {potential_keys}. Using alphanumeric sort.")
            sorted_layer_keys_to_use = sorted(potential_keys)
        if sorted_layer_keys_to_use:
            logging.info(f"IS Probe will use embeddings from layers in this order: {sorted_layer_keys_to_use}")
        else:
            logging.error("No valid layer embedding keys (e.g., 'layer_-1' with non-None tensor) found in the first sample's embedding dictionary.")
            return None, None, None
    else:
        logging.error("Could not find any valid dictionary-based embeddings in generations_data to determine layer structure for IS probe.")
        return None, None, None

    expected_embedding_dim_per_layer = None
    if sorted_layer_keys_to_use:
        try:
            first_emb_tensor_example = first_valid_embedding_dict[sorted_layer_keys_to_use[0]]
            if isinstance(first_emb_tensor_example, torch.Tensor):
                expected_embedding_dim_per_layer = first_emb_tensor_example.numel()
            elif isinstance(first_emb_tensor_example, (np.ndarray, list)):
                 expected_embedding_dim_per_layer = torch.tensor(first_emb_tensor_example).numel()
        except Exception as e_dim:
            logging.error(f"Could not determine embedding dimension: {e_dim}")
            return None, None, None


    for task_id, task_details in generations_data.items():
        try:
            most_likely = task_details.get("most_likely_answer", {})
            embedding_obj = most_likely.get("embedding")
            metric_score_raw = most_likely.get("accuracy_metric_score")

            if isinstance(embedding_obj, dict) and metric_score_raw is not None and not pd.isna(metric_score_raw):
                concatenated_embeddings_for_sample = []
                valid_sample_embeddings = True
                for layer_key in sorted_layer_keys_to_use:
                    layer_emb_tensor = embedding_obj.get(layer_key)

                    if layer_emb_tensor is not None:
                        try:
                            if isinstance(layer_emb_tensor, torch.Tensor):
                                current_emb = layer_emb_tensor.cpu().flatten()
                            else:
                                current_emb = torch.tensor(layer_emb_tensor).cpu().flatten()

                            if expected_embedding_dim_per_layer and current_emb.numel() != expected_embedding_dim_per_layer:
                                logging.warning(f"Task {task_id}, layer {layer_key}: Embedding dim mismatch. Expected {expected_embedding_dim_per_layer}, got {current_emb.numel()}. Filling with zeros.")
                                current_emb = torch.zeros(expected_embedding_dim_per_layer, dtype=torch.float32)
                            concatenated_embeddings_for_sample.append(current_emb)
                        except Exception as e_tensor_conv:
                            logging.warning(f"Task {task_id}, layer {layer_key}: Error processing embedding tensor: {e_tensor_conv}. Filling with zeros.")
                            if expected_embedding_dim_per_layer:
                                concatenated_embeddings_for_sample.append(torch.zeros(expected_embedding_dim_per_layer, dtype=torch.float32))
                            else:
                                valid_sample_embeddings = False; break
                    else:
                        logging.warning(f"Task {task_id}: Embedding for layer {layer_key} is None. Filling with zeros.")
                        if expected_embedding_dim_per_layer:
                             concatenated_embeddings_for_sample.append(torch.zeros(expected_embedding_dim_per_layer, dtype=torch.float32))
                        else:
                             valid_sample_embeddings = False; break

                if valid_sample_embeddings and concatenated_embeddings_for_sample:
                    final_concatenated_hs_tensor = torch.cat(concatenated_embeddings_for_sample)
                    hidden_states_list.append(final_concatenated_hs_tensor)
                    labels.append(metric_score_raw > metric_threshold)
                    task_ids.append(task_id)
                    processed_count += 1
                else:

                    logging.warning(f"Task {task_id}: Could not form a complete feature vector from layer embeddings. Skipping.")
                    skipped_count += 1

            elif embedding_obj is not None and not isinstance(embedding_obj, dict) and metric_score_raw is not None and not pd.isna(metric_score_raw):
                 if len(sorted_layer_keys_to_use) == 1:
                    try:
                        if isinstance(embedding_obj, torch.Tensor): hs_tensor = embedding_obj.cpu().flatten()
                        else: hs_tensor = torch.tensor(embedding_obj).cpu().flatten()

                        if expected_embedding_dim_per_layer and hs_tensor.numel() != expected_embedding_dim_per_layer:
                             logging.warning(f"Task {task_id} (old format): Dim mismatch. Expected {expected_embedding_dim_per_layer}, got {hs_tensor.numel()}. Filling.")
                             hs_tensor = torch.zeros(expected_embedding_dim_per_layer, dtype=torch.float32)

                        hidden_states_list.append(hs_tensor)
                        labels.append(metric_score_raw > metric_threshold)
                        task_ids.append(task_id)
                        processed_count += 1
                    except Exception as e_old_fmt:
                        logging.warning(f"Task {task_id}: Error processing old format embedding: {e_old_fmt}. Skipping.")
                        skipped_count += 1
                 else:
                    logging.warning(f"Task {task_id}: Non-dict embedding found but multiple layers expected by probe config. Skipping.")
                    skipped_count += 1
            else:
                logging.warning(f"Task {task_id} missing embedding object or metric score. Skipping.")
                skipped_count += 1
        except Exception as e:
            logging.error(f"Outer error processing task {task_id} for probe data: {e}", exc_info=True)
            skipped_count += 1

    logging.info(f"IS Probe data preparation complete. Processed: {processed_count}, Skipped: {skipped_count}, Total Attempted: {len(generations_data)}")
    if not hidden_states_list:
        logging.error("No valid hidden states extracted for the IS probe after multi-layer processing.")
        return None, None, None

    return hidden_states_list, labels, task_ids


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


def run_internal_signal_probe_cv(run_id, base_dir, classifier_type='logistic', n_splits=5, random_seed=42, metric_threshold=0.85):

    """
    Runs the Internal Signal probe with K-Fold Cross-Validation using the given classifier type.
    Also trains a final probe model on all valid data.

    Parameters
    ----------
    run_id : str
        Unique identifier for the generation run.
    base_dir : str
        Base directory containing the generation data and where output will be written.
    classifier_type : str
        One of 'logistic', 'svm', or 'random_forest'. Default is 'logistic'.
    n_splits : int
        Number of folds for K-Fold Cross-Validation. Default is 5.
    random_seed : int
        Random seed for K-Fold splitting. Default is 42.
    metric_threshold : float
        Threshold value for the metric (e.g., accuracy) to be used as the class label.
        Default is 0.85.

    Returns
    -------
    bool
        True if successful, False if an error occurred.
    """
    setup_logger()
    run_dir = Path(base_dir)
    logging.info(f"--- Running IS Probe K-Fold CV & Full Prediction for Run: {run_id} using {classifier_type} ---")
    generations_path = run_dir / "validation_generations.pkl"
    generations_data = load_pickle(generations_path)
    if generations_data is None: return False

    hidden_states, labels, task_ids_all = prepare_probe_data(generations_data, metric_threshold=metric_threshold)

    if hidden_states is None: return False
    if len(np.unique(labels)) < 2: logging.error("Only one class label found. Cannot train/evaluate probe."); return False

    logging.info(f"Prepared {len(hidden_states)} embeddings and labels for K-Fold CV.")

    try:
        X_full = np.array([vec.numpy() for vec in hidden_states])
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
    parser.add_argument("--metric_threshold", type=float, required=True, help="Metric threshold defining correctness.")

    args = parser.parse_args()

    run_internal_signal_probe_cv(
        args.run_id,
        args.base_dir,
        classifier_type=args.classifier,
        n_splits=args.n_splits,
        random_seed=args.seed,
        metric_threshold=args.metric_threshold
    )