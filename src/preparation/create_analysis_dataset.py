import pandas as pd
import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    if not filepath.is_file(): logging.error(f"Pickle file not found: {filepath}"); return None
    try:
        with open(filepath, "rb") as f: data = pickle.load(f)
        logging.info(f"Successfully loaded {filepath}"); return data
    except Exception as e: logging.error(f"Error loading pickle file {filepath}: {e}", exc_info=True); return None


def infer_type(subtype):
    """
    Infers the main type category from a given subtype string.

    Args:
        subtype: The subtype string to be analyzed. Can be NaN or a string identifying error types.

    Returns:
        A string indicating the inferred type category, which can be:
        - 'Factuality' for factuality-related subtypes.
        - 'Faithfulness' for faithfulness-related subtypes.
        - 'Other/Unclear' for explicitly marked unclear types.
        - 'Annotation_Failed' for annotation failures.
        - 'Unknown' if the subtype cannot be inferred.

    Raises:
        Logs a warning if the subtype cannot be inferred.
    """

    if pd.isna(subtype): return pd.NA
    subtype_str = str(subtype).strip()
    if subtype_str.startswith('A1_Dosage') or subtype_str.startswith('A2_Contra') or \
       subtype_str.startswith('A3_Diagn') or subtype_str.startswith('B1_Statis') or \
       subtype_str.startswith('B2_Fabri') or subtype_str == 'Factuality_Other': return 'Factuality'
    elif subtype_str.startswith('A1_Context') or subtype_str.startswith('A2_Instruc') or \
         subtype_str.startswith('B1_Extrap') or subtype_str == 'Faithfulness_Other': return 'Faithfulness'
    elif subtype_str == 'Other/Unclear': return 'Other/Unclear'
    elif "[API_FAILED]" in subtype_str or "[BLOCKED" in subtype_str: return "Annotation_Failed"
    else: logging.warning(f"Could not infer type for subtype: '{subtype_str}'"); return 'Unknown'


def load_reviewed_annotations(csv_path):
    """
    Loads a CSV file containing reviewed annotations into a Pandas DataFrame. The CSV must contain columns for 'id', 'automated_subtype', and 'automated_rationale'. The function renames the subtype column to 'hallucination_subtype' and the rationale column to 'annotation_rationale'. It also infers the main type category ('Factuality', 'Faithfulness', 'Other/Unclear', 'Annotation_Failed', 'Unknown') from the subtype string and adds it as a column named 'hallucination_type'. The function logs errors and warnings during processing, and returns None if there is an error.

    Parameters
    ----------
    csv_path : str
        The path to the CSV containing the reviewed annotations.

    Returns
    -------
    df : pd.DataFrame
        The loaded DataFrame with the specified columns and inferred 'hallucination_type'.
    """
    path = Path(csv_path)
    if not path.is_file(): logging.error(f"Annotations CSV not found: {path}"); return None
    try:
        df = pd.read_csv(path); logging.info(f"Loaded {len(df)} reviewed annotations from {path}")
        id_col, subtype_col, rationale_col = 'id', 'automated_subtype', 'automated_rationale'
        required_cols = [id_col, subtype_col]
        if not all(c in df.columns for c in required_cols): logging.error(f"Annotations CSV missing {required_cols}. Found: {df.columns.tolist()}"); return None
        df.rename(columns={subtype_col: 'hallucination_subtype', rationale_col: 'annotation_rationale'}, inplace=True)
        df['hallucination_type'] = df['hallucination_subtype'].apply(infer_type)
        cols_to_keep = ['id', 'hallucination_subtype', 'hallucination_type']
        if 'annotation_rationale' in df.columns: cols_to_keep.append('annotation_rationale')
        df['id'] = df['id'].astype(str).str.strip(); return df[cols_to_keep]
    except Exception as e: logging.error(f"Error processing reviewed annotations CSV {path}: {e}"); return None


def load_score_csv(csv_path, id_col='id', score_col_name='score'):
    """
    Loads scores from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing scores.
    id_col : str, optional
        The name of the column containing unique identifiers (default is 'id').
    score_col_name : str, optional
        The name of the column containing score values (default is 'score').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the 'id' and 'score' columns. If the file is not found
        or there is an error, returns an empty DataFrame with the specified columns.
        Logs warnings and errors as needed.
    """

    path = Path(csv_path)
    if not path.is_file(): logging.warning(f"Score CSV not found: {path}. Scores for '{score_col_name}' missing."); return pd.DataFrame(columns=[id_col, score_col_name])
    try:
        df = pd.read_csv(path); logging.info(f"Loaded {len(df)} scores for '{score_col_name}' from {path}")
        if id_col not in df.columns or score_col_name not in df.columns: logging.error(f"CSV {path} missing '{id_col}' or '{score_col_name}'."); return pd.DataFrame(columns=[id_col, score_col_name])
        df[id_col] = df[id_col].astype(str).str.strip(); return df[[id_col, score_col_name]]
    except Exception as e: logging.error(f"Error loading score CSV {path}: {e}"); return pd.DataFrame(columns=[id_col, score_col_name])


def load_se_scores_from_pkl(run_dir_path):
    """
    Loads semantic entropy (SE) scores, naive entropy scores, and automated correctness labels from PKL files in a run directory.

    Parameters
    ----------
    run_dir_path : str
        The path to the run directory containing the uncertainty_measures.pkl and validation_generations.pkl files.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['id', 'semantic_entropy', 'naive_entropy', 'automated_is_correct'].
        If there is an error, returns None and logs the error.
    """
    run_dir = Path(run_dir_path); uncertainty_path = run_dir / "uncertainty_measures.pkl"; generations_path = run_dir / "validation_generations.pkl"
    uncertainty_data = load_pickle(uncertainty_path); generations_data = load_pickle(generations_path)
    if uncertainty_data is None or generations_data is None: logging.error("Need pkls for SE scores."); return None
    logging.info("Loading SE/Naive scores and automated correctness labels...");
    try:
        measures = uncertainty_data.get("uncertainty_measures", {}); se_scores = measures.get("semantic_entropy"); naive_entropy = measures.get("regular_entropy"); validation_is_false = uncertainty_data.get("validation_is_false")
        task_ids_ordered = list(generations_data.keys()); num_tasks = len(task_ids_ordered)
        def pad_or_truncate(data_list, name, target_len):
            if data_list is None: return [np.nan] * target_len
            current_len = len(data_list);
            if current_len < target_len: return data_list + [np.nan] * (target_len - current_len)
            elif current_len > target_len: return data_list[:target_len]
            return data_list
        se_scores = pad_or_truncate(se_scores, 'semantic_entropy', num_tasks); naive_entropy = pad_or_truncate(naive_entropy, 'regular_entropy', num_tasks); validation_is_false = pad_or_truncate(validation_is_false, 'validation_is_false', num_tasks)
        data_dict = {"id": [str(tid).strip() for tid in task_ids_ordered], "semantic_entropy": se_scores, "naive_entropy": naive_entropy, "automated_is_correct": [np.nan if pd.isna(x) else not x for x in validation_is_false]}
        scores_df = pd.DataFrame(data_dict); logging.info(f"Loaded SE/Naive/AutoCorrect scores for {len(scores_df)} tasks."); return scores_df
    except Exception as e: logging.error(f"Error loading SE/Naive from pkl: {e}"); return None


def load_generation_details(pkl_path, accuracy_threshold=0.5):
    """
    Loads generation details from a pickle file into a DataFrame with columns ['id', 'question', 'context', 'reference_answers', 'generated_answer', 'accuracy_metric_score', 'is_correct_auto_from_gen'].

    Parameters
    ----------
    pkl_path : str
        The path to the pickle file containing the generation details.
    accuracy_threshold : float, default=0.5
        The accuracy threshold above which a generation is considered correct.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ['id', 'question', 'context', 'reference_answers', 'generated_answer', 'accuracy_metric_score', 'is_correct_auto_from_gen'].
        If there is an error, returns None and logs the error.
    """
    generations_data = load_pickle(pkl_path);
    if generations_data is None: return None
    logging.info("Loading generation details..."); data_list = []
    for task_id, task_details in generations_data.items():
        most_likely = task_details.get("most_likely_answer", {}); reference = task_details.get("reference", {}); answers = reference.get('answers', {}); ref_texts = answers.get('text', [])
        accuracy = most_likely.get("accuracy"); is_correct_auto = accuracy > accuracy_threshold if accuracy is not None else None
        data_list.append({"id": str(task_id).strip(), "question": task_details.get("question", ""), "context": task_details.get("context", None),
                          "reference_answers": str(ref_texts), "generated_answer": most_likely.get("response", "[GENERATION FAILED]"),
                          "accuracy_metric_score": most_likely.get("accuracy", np.nan), "is_correct_auto_from_gen": is_correct_auto})
    df = pd.DataFrame(data_list); logging.info(f"Loaded generation details for {len(df)} tasks."); return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge reviewed annotations with ALL scores for SUBTYPE analysis.")
    parser.add_argument("reviewed_annotations_csv", help="Path to reviewed automated annotations CSV.")
    parser.add_argument("run_files_dir", help="Path to run files directory.")
    parser.add_argument("--is_scores_all_csv", required=True, help="Path to CSV with IS scores for ALL samples.")
    parser.add_argument("--hybrid_meta_scores_all_csv", required=True, help="Path to CSV with Hybrid Meta scores for ALL samples.")
    parser.add_argument("final_output_csv", help="Path to save final merged CSV for subtype analysis.")
    args = parser.parse_args()

    df_annotations = load_reviewed_annotations(args.reviewed_annotations_csv)
    df_se_naive_auto = load_se_scores_from_pkl(args.run_files_dir)
    df_is_scores = load_score_csv(args.is_scores_all_csv, score_col_name='internal_signal_score')
    df_hybrid_meta = load_score_csv(args.hybrid_meta_scores_all_csv, score_col_name='hybrid_meta_score')
    gen_pkl_path = Path(args.run_files_dir) / "validation_generations.pkl"
    df_generation_details = load_generation_details(gen_pkl_path)

    if any(df is None for df in [df_annotations, df_se_naive_auto, df_generation_details]) or \
       (df_is_scores.empty and Path(args.is_scores_all_csv).exists()) or \
       (df_hybrid_meta.empty and Path(args.hybrid_meta_scores_all_csv).exists()):
        logging.error("One or more essential input files failed to load. Exiting."); exit(1)

    final_df = df_annotations.copy(); logging.info(f"Starting merge with {len(final_df)} annotated samples.")
    final_df = pd.merge(final_df, df_se_naive_auto, on="id", how="left")
    final_df = pd.merge(final_df, df_is_scores, on="id", how="left")
    final_df = pd.merge(final_df, df_hybrid_meta, on="id", how="left")
    cols_to_merge = ['id'] + [col for col in df_generation_details.columns if col not in final_df.columns and col != 'id']
    final_df = pd.merge(final_df, df_generation_details[cols_to_merge], on="id", how="left")
    logging.info(f"Annotated subset row count after merging: {len(final_df)}")

    se_col, is_col, hybrid_simple_col = 'semantic_entropy', 'internal_signal_score', 'hybrid_simple_score'
    if se_col in final_df.columns and is_col in final_df.columns:
        logging.info("Calculating simple hybrid score for the annotated subset...")
        scaler_se, scaler_is = MinMaxScaler(), MinMaxScaler()
        se_subset, is_subset = final_df[se_col].dropna(), final_df[is_col].dropna()
        se_norm_col, is_norm_col = f"{se_col}_norm_subset", f"{is_col}_norm_subset"
        final_df[se_norm_col], final_df[is_norm_col] = np.nan, np.nan
        if not se_subset.empty:
            try: final_df.loc[se_subset.index, se_norm_col] = scaler_se.fit_transform(se_subset.values.reshape(-1, 1)).flatten()
            except ValueError: logging.warning("SE constant in subset."); final_df.loc[se_subset.index, se_norm_col] = 0.5
        if not is_subset.empty:
            try: final_df.loc[is_subset.index, is_norm_col] = scaler_is.fit_transform(is_subset.values.reshape(-1, 1)).flatten()
            except ValueError: logging.warning("IS constant in subset."); final_df.loc[is_subset.index, is_norm_col] = 0.5
        final_df[hybrid_simple_col] = 0.5 * final_df[se_norm_col] + 0.5 * final_df[is_norm_col]
        logging.info(f"Calculated '{hybrid_simple_col}'. Missing: {final_df[hybrid_simple_col].isnull().sum()}")
    else: final_df[hybrid_simple_col] = np.nan

    logging.info("Final DataFrame info for subtype analysis:")
    final_df.info()
    final_cols = [c for c in [
        'id', 'question', 'context', 'reference_answers', 'generated_answer',
        'accuracy_metric_score', 'automated_is_correct',
        'hallucination_type', 'hallucination_subtype', 'annotation_rationale',
        'semantic_entropy', 'naive_entropy', 'internal_signal_score',
        hybrid_simple_col, 'hybrid_meta_score'
    ] if c in final_df.columns]
    final_df_output = final_df[final_cols]

    final_path = Path(args.final_output_csv); final_path.parent.mkdir(parents=True, exist_ok=True)
    try: final_df_output.to_csv(final_path, index=False); logging.info(f"Final subtype analysis data ({len(final_df_output)} samples) saved to: {final_path}")
    except IOError as e: logging.error(f"Error writing final file {final_path}: {e}")
    logging.info("\n--- Data Preparation Complete ---")