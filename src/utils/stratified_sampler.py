import argparse
import pickle
import logging
import random
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pickle(filepath):
    """
    Loads a pickle file. If the file is not found or there is an error, return None.

    Parameters
    ----------
    filepath : str
        The path to the pickle file.

    Returns
    -------
    data : object
        The loaded pickle file.
    """
    filepath = Path(filepath)
    if not filepath.is_file():
        logging.error(f"Pickle file not found: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file {filepath}: {e}")
        return None

def match_scores_to_generations(generations_data, uncertainty_data):
    """
    Matches semantic entropy scores to generation data and returns a DataFrame.

    This function takes generation data and uncertainty data, specifically
    semantic entropy scores, and matches them based on task IDs. It validates
    the data consistency and logs appropriate warnings or errors when
    mismatches or missing data are found.

    Parameters
    ----------
    generations_data : dict
        A dictionary where keys are task IDs and values are details of the
        respective generations, including the 'most_likely_answer' and its
        'accuracy'.
    uncertainty_data : dict
        A dictionary containing uncertainty measures, specifically the
        'semantic_entropy' scores.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with columns ['id', 'is_correct', 'accuracy_score',
        'semantic_entropy'] containing matched data for each task. Returns None
        if no data could be matched or if input data is missing or improperly
        formatted.
    """

    if not generations_data or not uncertainty_data:
        logging.error("Generations or uncertainty data is missing.")
        return None

    try:
        se_scores = uncertainty_data['uncertainty_measures']['semantic_entropy']
    except KeyError as e:
        logging.error(f"Missing expected key in uncertainty_data: {e}")
        return None

    num_generations = len(generations_data)
    num_se_scores = len(se_scores)

    if num_generations != num_se_scores:
        logging.warning(f"Mismatch between number of generations ({num_generations}) and SE scores ({num_se_scores}). Assuming scores correspond to the first {min(num_generations, num_se_scores)} generations processed in order.")

    matched_data = []
    for i, task_id in enumerate(generations_data.keys()):
        if i >= num_se_scores:
             logging.warning(f"No SE score found for generation index {i} (task_id {task_id}). Skipping.")
             continue

        task_details = generations_data[task_id]
        try:
            most_likely = task_details.get('most_likely_answer', {})
            accuracy = most_likely.get('accuracy', 0.0)
            is_correct = accuracy > 0.5
            semantic_entropy = se_scores[i]

            matched_data.append({
                'id': task_id,
                'is_correct': is_correct,
                'accuracy_score': accuracy,
                'semantic_entropy': semantic_entropy
            })
        except KeyError as e:
             logging.warning(f"Missing key for task_id {task_id}: {e}. Skipping task.")
             continue
        except IndexError:
             logging.error(f"IndexError while accessing score for index {i}. Score list length: {num_se_scores}")
             continue


    if not matched_data:
        logging.error("No data could be matched.")
        return None

    return pd.DataFrame(matched_data)


def sample_for_annotation(run_id, base_dir, num_bins, samples_per_bin, output_file, random_seed):
    """
    Sample incorrect validation samples for annotation, stratified by semantic entropy.

    Parameters
    ----------
    run_id : str
        The ID of the run.
    base_dir : str
        The base directory where run files are stored (e.g., './wandb/run-<run_id>/files' or a custom path).
    num_bins : int, default=3
        The number of bins (strata) to divide SE scores into (e.g., 3 for terciles).
    samples_per_bin : int, default=40
        The number of samples to draw from each bin.
    output_file : str, default="sampled_ids_for_annotation.txt"
        The name of the output file to save the list of sampled task IDs.
    random_seed : int, default=42
        The random seed for reproducibility.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    run_dir = Path(base_dir)

    logging.info(f"Processing run: {run_id}")
    logging.info(f"Looking for files in: {run_dir}")

    generations_path = run_dir / 'validation_generations.pkl'
    uncertainty_path = run_dir / 'uncertainty_measures.pkl'

    generations_data = load_pickle(generations_path)
    uncertainty_data = load_pickle(uncertainty_path)

    if generations_data is None or uncertainty_data is None:
        logging.error("Failed to load necessary data files. Exiting.")
        return

    matched_df = match_scores_to_generations(generations_data, uncertainty_data)
    if matched_df is None:
        logging.error("Failed to match scores to generations. Exiting.")
        return

    incorrect_df = matched_df[~matched_df['is_correct']].copy()
    num_incorrect = len(incorrect_df)
    logging.info(f"Found {num_incorrect} incorrect samples.")

    if num_incorrect == 0:
        logging.warning("No incorrect samples found for this run. No IDs to sample.")
        with open(output_file, 'w') as f:
            f.write("")
        return

    try:
        incorrect_df['se_bin'] = pd.qcut(incorrect_df['semantic_entropy'], num_bins, labels=False, duplicates='drop')
        logging.info(f"Stratified incorrect samples into {num_bins} bins based on SE.")
        logging.info(f"\nBin counts:\n{incorrect_df['se_bin'].value_counts().sort_index()}")
    except ValueError as e:
        logging.error(f"Error during stratification (maybe too few unique SE values or samples?): {e}")
        logging.info("Attempting stratification with pd.cut instead.")
        try:
             incorrect_df['se_bin'] = pd.cut(incorrect_df['semantic_entropy'], num_bins, labels=False, include_lowest=True)
             logging.info(f"Stratified incorrect samples using equal-width bins.")
             logging.info(f"\nBin counts:\n{incorrect_df['se_bin'].value_counts().sort_index()}")
        except Exception as cut_e:
             logging.error(f"Error during pd.cut stratification as well: {cut_e}")
             logging.error("Cannot perform stratification. Exiting.")
             return


    sampled_ids = []
    for bin_label in range(num_bins):
        bin_df = incorrect_df[incorrect_df['se_bin'] == bin_label]
        n_in_bin = len(bin_df)
        logging.info(f"Processing Bin {bin_label}: {n_in_bin} samples available.")

        if n_in_bin == 0:
            logging.warning(f"Bin {bin_label} is empty. Skipping.")
            continue

        n_to_sample = min(samples_per_bin, n_in_bin)
        if n_in_bin < samples_per_bin:
            logging.warning(f"Bin {bin_label} has only {n_in_bin} samples, sampling all of them (requested {samples_per_bin}).")

        sampled_bin_ids = bin_df['id'].sample(n=n_to_sample, random_state=random_seed).tolist()
        sampled_ids.extend(sampled_bin_ids)
        logging.info(f"Sampled {len(sampled_bin_ids)} IDs from Bin {bin_label}.")

    logging.info(f"Total samples selected for annotation: {len(sampled_ids)}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            for task_id in sampled_ids:
                f.write(f"{task_id}\n")
        logging.info(f"List of sampled IDs saved to: {output_path}")

    except IOError as e:
        logging.error(f"Error writing output file {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified sampling of incorrect validation samples based on Semantic Entropy for annotation.")
    parser.add_argument("run_id", type=str, help="WandB Run ID of the experiment to process.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory where run files are stored (e.g., './wandb/run-<run_id>/files' or a custom path).")
    parser.add_argument("--num_bins", type=int, default=3, help="Number of bins (strata) to divide SE scores into (e.g., 3 for terciles).")
    parser.add_argument("--samples_per_bin", type=int, default=40, help="Number of samples to draw from each bin.")
    parser.add_argument("--output_file", type=str, default="sampled_ids_for_annotation.txt", help="Name of the output file to save the list of sampled task IDs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    output_dir = Path(f"./annotation_data/{args.run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    full_output_path = output_dir / args.output_file

    sample_for_annotation(
        run_id=args.run_id,
        base_dir=args.base_dir,
        num_bins=args.num_bins,
        samples_per_bin=args.samples_per_bin,
        output_file=full_output_path,
        random_seed=args.seed
    )