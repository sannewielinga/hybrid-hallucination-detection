import argparse
import pickle
import logging
import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_pickle(filepath):
    """Loads a pickle file."""
    filepath = Path(filepath)
    if not filepath.is_file():
        logging.error(f"Pickle file not found: {filepath}")
        return None
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file {filepath}: {e}")
        return None


def prepare_probe_data(generations_data, accuracy_threshold=0.5):
    """Extracts hidden states and correctness labels for probing."""
    hidden_states = []
    labels = []
    task_ids = []

    logging.info("Preparing data for internal signal probe...")
    processed_count = 0
    skipped_count = 0

    for task_id, task_details in generations_data.items():
        try:
            most_likely = task_details.get("most_likely_answer", {})
            embedding = most_likely.get("embedding")
            accuracy = most_likely.get("accuracy")

            if embedding is not None and accuracy is not None:
                if isinstance(embedding, torch.Tensor):
                    hidden_states.append(embedding.cpu())
                else:
                    try:
                        hidden_states.append(torch.tensor(embedding).cpu())
                    except Exception as conversion_e:
                        logging.warning(
                            f"Could not convert embedding for task {task_id} to tensor: {conversion_e}. Skipping."
                        )
                        skipped_count += 1
                        continue

                is_correct = accuracy > accuracy_threshold
                labels.append(is_correct)
                task_ids.append(task_id)
                processed_count += 1
            else:
                logging.warning(
                    f"Task {task_id} missing 'embedding' or 'accuracy' in 'most_likely_answer'. Skipping."
                )
                skipped_count += 1

        except Exception as e:
            logging.error(f"Error processing task {task_id}: {e}. Skipping.")
            skipped_count += 1

    logging.info(
        f"Data preparation complete. Processed: {processed_count}, Skipped: {skipped_count}"
    )
    if not hidden_states:
        logging.error("No valid hidden states extracted.")
        return None, None, None

    return hidden_states, labels, task_ids


def load_se_scores(run_dir_path, task_ids_ordered):
    """Loads SE scores ensuring order matches task_ids_ordered."""
    run_dir = Path(run_dir_path)
    uncertainty_path = run_dir / "uncertainty_measures.pkl"

    uncertainty_data = load_pickle(uncertainty_path)
    if uncertainty_data is None:
        return None

    logging.info("Loading SE scores...")
    try:
        uncertainty_measures = uncertainty_data.get("uncertainty_measures", {})
        se_scores_list = uncertainty_measures.get("semantic_entropy")
        if se_scores_list is None:
            logging.error("Missing 'semantic_entropy' in uncertainty_measures.pkl")
            return None

        if len(task_ids_ordered) != len(se_scores_list):
            logging.warning(
                f"Mismatch length: {len(task_ids_ordered)} tasks vs {len(se_scores_list)} SE scores. Using min length."
            )
            min_len = min(len(task_ids_ordered), len(se_scores_list))
            id_to_score_map = {
                task_id: se_scores_list[i]
                for i, task_id in enumerate(task_ids_ordered[:min_len])
            }
        else:
            id_to_score_map = {
                task_id: se_scores_list[i] for i, task_id in enumerate(task_ids_ordered)
            }

        logging.info(f"SE scores loaded for {len(id_to_score_map)} tasks.")
        return id_to_score_map

    except Exception as e:
        logging.error(f"Error loading SE scores: {e}")
        return None


def run_internal_signal_probe(run_id, base_dir, test_size, random_seed, output_csv):
    """Loads data, trains probe, evaluates, calculates hybrid scores, and saves results."""
    run_dir = Path(base_dir)
    logging.info(f"--- Starting Internal Signal Probe for Run: {run_id} ---")
    logging.info(f"Looking for files in: {run_dir}")

    generations_path = run_dir / "validation_generations.pkl"
    generations_data = load_pickle(generations_path)
    if generations_data is None:
        return

    hidden_states, labels, task_ids_all = prepare_probe_data(generations_data)
    if hidden_states is None:
        return

    if len(np.unique(labels)) < 2:
        logging.error(
            "Only one class found in labels. Cannot train classifier. Check accuracy threshold or data."
        )
        return

    try:
        hs_train, hs_test, y_train, y_test, ids_train, ids_test = train_test_split(
            hidden_states,
            labels,
            task_ids_all,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels,
        )
        logging.info(f"Split data: {len(hs_train)} train, {len(hs_test)} test samples.")
    except ValueError as e:
        logging.error(f"Error during train/test split (perhaps too few samples?): {e}")
        return

    logging.info("Training Logistic Regression probe...")
    try:
        X_train = np.array([vec.numpy().flatten() for vec in hs_train])
        y_train = np.array(y_train)

        if len(np.unique(y_train)) < 2:
            logging.error(
                "Training set contains only one class after split. Cannot train."
            )
            return

        classifier = LogisticRegression(
            random_state=random_seed, class_weight="balanced", max_iter=1000
        )
        classifier.fit(X_train, y_train)
        logging.info("Probe training complete.")
    except Exception as e:
        logging.error(f"Error during classifier training: {e}")
        return

    logging.info("Evaluating probe on test set...")
    try:
        X_test = np.array([vec.numpy().flatten() for vec in hs_test])
        y_test = np.array(y_test)

        if len(np.unique(y_test)) < 2:
            logging.warning(
                "Test set contains only one class after split. Some metrics might be undefined."
            )
            accuracy = classifier.score(X_test, y_test)
            auc = np.nan
            probabilities = classifier.predict_proba(X_test)
        else:
            accuracy = classifier.score(X_test, y_test)
            probabilities = classifier.predict_proba(X_test)
            false_class_index = classifier.classes_.tolist().index(False)
            internal_signal_scores = probabilities[:, false_class_index]
            auc = roc_auc_score(y_test == False, internal_signal_scores)

        logging.info(f"Internal Signal Probe Performance:")
        logging.info(f"  Accuracy: {accuracy:.4f}")
        logging.info(f"  AUC (Predicting Incorrect): {auc:.4f}")

        results_df = pd.DataFrame(
            {
                "id": ids_test,
                "is_correct": y_test,
                "internal_signal_score": (
                    internal_signal_scores if len(np.unique(y_test)) == 2 else np.nan
                ),
            }
        )

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        return

    all_task_ids_ordered = list(generations_data.keys())
    id_to_se_score_map = load_se_scores(run_dir, all_task_ids_ordered)
    if id_to_se_score_map is None:
        logging.warning("Could not load SE scores. Hybrid analysis will be skipped.")
        se_scores_test = pd.Series([np.nan] * len(results_df), index=results_df.index)
    else:
        results_df["semantic_entropy"] = results_df["id"].map(id_to_se_score_map)
        if results_df["semantic_entropy"].isnull().any():
            logging.warning(
                "Some test samples could not be matched with SE scores. Check ID consistency."
            )

    logging.info("Performing hybridization analysis...")
    if (
        "semantic_entropy" in results_df.columns
        and "internal_signal_score" in results_df.columns
        and not results_df["semantic_entropy"].isnull().all()
        and not results_df["internal_signal_score"].isnull().all()
    ):

        valid_scores_df = results_df.dropna(
            subset=["semantic_entropy", "internal_signal_score"]
        ).copy()
        logging.info(
            f"Analyzing {len(valid_scores_df)} samples with valid SE and IS scores."
        )

        if (
            len(valid_scores_df) > 1
            and len(valid_scores_df["is_correct"].unique()) == 2
        ):
            scaler_se = MinMaxScaler()
            scaler_is = MinMaxScaler()
            try:
                valid_scores_df["se_normalized"] = scaler_se.fit_transform(
                    valid_scores_df[["semantic_entropy"]]
                )
            except ValueError:
                logging.warning(
                    "SE scores might be constant, cannot normalize with MinMaxScaler."
                )
                valid_scores_df["se_normalized"] = 0.5
            try:
                valid_scores_df["is_normalized"] = scaler_is.fit_transform(
                    valid_scores_df[["internal_signal_score"]]
                )
            except ValueError:
                logging.warning(
                    "IS scores might be constant, cannot normalize with MinMaxScaler."
                )
                valid_scores_df["is_normalized"] = 0.5

            valid_scores_df["hybrid_0.5_0.5"] = (
                0.5 * valid_scores_df["se_normalized"]
                + 0.5 * valid_scores_df["is_normalized"]
            )

            true_incorrect_labels = valid_scores_df["is_correct"] == False
            auc_se_only = roc_auc_score(
                true_incorrect_labels, valid_scores_df["semantic_entropy"]
            )
            auc_is_only = roc_auc_score(
                true_incorrect_labels, valid_scores_df["internal_signal_score"]
            )
            auc_hybrid_05 = roc_auc_score(
                true_incorrect_labels, valid_scores_df["hybrid_0.5_0.5"]
            )

            logging.info("--- Hybridization AUC Results (on Test Set) ---")
            logging.info(f"  SE Only AUC:              {auc_se_only:.4f}")
            logging.info(f"  Internal Signal Only AUC: {auc_is_only:.4f}")
            logging.info(f"  Hybrid (0.5/0.5) AUC:   {auc_hybrid_05:.4f}")

            results_df = results_df.merge(
                valid_scores_df[
                    ["id", "se_normalized", "is_normalized", "hybrid_0.5_0.5"]
                ],
                on="id",
                how="left",
            )

        else:
            logging.warning(
                "Not enough data or classes after dropping NaNs to calculate hybrid AUCs."
            )
    else:
        logging.warning(
            "Skipping hybridization analysis as SE or IS scores are missing/invalid."
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df.to_csv(output_path, index=False)
        logging.info(f"Detailed test set results saved to: {output_path}")
    except IOError as e:
        logging.error(f"Error writing output file {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate an internal signal probe for hallucination detection."
    )
    parser.add_argument(
        "run_id", type=str, help="WandB Run ID of the experiment to process."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Directory containing run files ('validation_generations.pkl', 'uncertainty_measures.pkl').",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Proportion of data to use for the test set for the probe.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="internal_signal_results.csv",
        help="Name for the output CSV file saving test set results.",
    )

    args = parser.parse_args()

    full_output_path = args.output_csv

    run_internal_signal_probe(
        run_id=args.run_id,
        base_dir=args.base_dir,
        test_size=args.test_size,
        random_seed=args.seed,
        output_csv=full_output_path,
    )
