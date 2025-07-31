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
    if not filepath.is_file():
        logging.error(f"Pickle file not found: {filepath}")
        return None
    try:
        with open(filepath, "rb") as f: data = pickle.load(f)
        logging.info(f"Successfully loaded {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file {filepath}: {e}", exc_info=True)
        return None

def infer_type(subtype):
    """
    Infers the type of a given subtype string.

    Parameters
    ----------
    subtype : str
        The subtype string to infer the type from.

    Returns
    -------
    type : str
        The inferred type, one of 'Factuality', 'Faithfulness', 'Other/Unclear', or 'Unknown' if the subtype does not match any known types.

    Notes
    -----
    This function assumes that the subtype string is one of the following:

    - A1_Dosage* (e.g., A1_DosageAdministration)
    - A2_Contra* (e.g., A2_ContraindicationIndication)
    - A3_Diagn* (e.g., A3_DiagnosticCriteriaDefinition)
    - B1_Statis* (e.g., B1_StatisticalEpidemiological)
    - B2_Fabri* (e.g., B2_FabricatedEntityGuideline)
    - Factuality_Other
    - A1_Context* (e.g., A1_ContextIgnorant)
    - A2_Instruc* (e.g., A2_InstructionMisinterpretation)
    - B1_Extrap* (e.g., B1_ExtrapolationAddition)
    - Faithfulness_Other
    - Other/Unclear
    - Factuality
    - Faithfulness

    If the subtype string does not match any of these patterns, the function returns 'Unknown' and logs a warning.
    """
    if pd.isna(subtype): return pd.NA
    subtype_str = str(subtype).strip()
    if subtype_str.startswith('A1_Dosage') or subtype_str.startswith('A2_Contra') or \
       subtype_str.startswith('A3_Diagn') or subtype_str.startswith('B1_Statis') or \
       subtype_str.startswith('B2_Fabri') or subtype_str == 'Factuality_Other':
        return 'Factuality'
    elif subtype_str.startswith('A1_Context') or subtype_str.startswith('A2_Instruc') or \
         subtype_str.startswith('B1_Extrap') or subtype_str == 'Faithfulness_Other':
        return 'Faithfulness'
    elif subtype_str == 'Other/Unclear':
        return 'Other/Unclear'
    elif subtype_str == 'Factuality':
        return 'Factuality'
    elif subtype_str == 'Faithfulness':
        return 'Faithfulness'
    else:
        logging.warning(f"Could not infer type for subtype: '{subtype_str}'")
        return 'Unknown'

def load_reviewed_annotations(csv_path):
    """
    Loads a CSV containing reviewed annotations and processes it into a DataFrame
    that can be joined with the uncertainty scores.

    The function will attempt to determine whether the CSV contains manual or
    automated annotations, and will use the appropriate column names to
    establish the 'hallucination_subtype' and 'hallucination_type' columns.

    If the CSV is missing required columns, or if there is an error processing
    the file, the function will return None.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file containing the reviewed annotations.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the reviewed annotations, with columns 'id',
        'hallucination_subtype', and 'hallucination_type'. If the CSV contained
        a 'annotation_rationale' column, it will be preserved in the output
        DataFrame.
    """
    path = Path(csv_path)
    if not path.is_file():
        logging.error(f"Annotations CSV not found: {path}"); return None
    try:
        df = pd.read_csv(path); logging.info(f"Loaded {len(df)} reviewed annotations from {path}")
        id_col, subtype_col_auto, rationale_col_auto = 'id', 'automated_subtype', 'automated_rationale'
        subtype_col_manual, rationale_col_manual = 'hallucination_subtype', 'annotation_rationale'

        if id_col not in df.columns:
            logging.error(f"Annotations CSV missing required '{id_col}' column. Found: {df.columns.tolist()}")
            return None

        if subtype_col_manual in df.columns and not df[subtype_col_manual].isnull().all():
            logging.info(f"Using manual annotation columns ('{subtype_col_manual}', '{rationale_col_manual if rationale_col_manual in df.columns else 'N/A'}').")
            df.rename(columns={subtype_col_manual: 'hallucination_subtype', 
                               rationale_col_manual: 'annotation_rationale' if rationale_col_manual in df.columns else 'annotation_rationale_placeholder'}, 
                      inplace=True)
        elif subtype_col_auto in df.columns:
            logging.info(f"Using automated annotation columns ('{subtype_col_auto}', '{rationale_col_auto if rationale_col_auto in df.columns else 'N/A'}').")
            df.rename(columns={subtype_col_auto: 'hallucination_subtype', 
                               rationale_col_auto: 'annotation_rationale' if rationale_col_auto in df.columns else 'annotation_rationale_placeholder'}, 
                      inplace=True)
        else:
            logging.error(f"Annotations CSV missing required subtype column (neither '{subtype_col_manual}' nor '{subtype_col_auto}' found with data). Found: {df.columns.tolist()}")
            return None
        
        if 'hallucination_subtype' not in df.columns:
             logging.error("Critical error: 'hallucination_subtype' column not established.")
             return None

        df['hallucination_type'] = df['hallucination_subtype'].apply(infer_type)
        cols_to_keep = ['id', 'hallucination_subtype', 'hallucination_type']
        if 'annotation_rationale' in df.columns:
            cols_to_keep.append('annotation_rationale')
        elif 'annotation_rationale_placeholder' in df.columns and 'annotation_rationale' not in cols_to_keep:
            df.rename(columns={'annotation_rationale_placeholder': 'annotation_rationale'}, inplace=True)
            cols_to_keep.append('annotation_rationale')


        df['id'] = df['id'].astype(str).str.strip()
        return df[cols_to_keep]
    except Exception as e:
        logging.error(f"Error processing reviewed annotations CSV {path}: {e}", exc_info=True)
        return None

def load_score_csv(csv_path, id_col='id', score_col_name='score'):
    """
    Load a CSV file containing uncertainty scores and return them as a DataFrame with columns 'id' and 'score'.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the scores
    id_col : str, optional
        Name of the column containing the sample IDs (default: 'id')
    score_col_name : str, optional
        Name of the column containing the uncertainty scores (default: 'score')
    
    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the scores, with columns 'id' and 'score'. If the CSV file is not found, an empty DataFrame with the same columns is returned.
    """
    path = Path(csv_path)
    if not path.is_file():
        logging.warning(f"Score CSV not found: {path}. Scores for '{score_col_name}' will be missing.")
        return pd.DataFrame(columns=[id_col, score_col_name])
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded {len(df)} scores for '{score_col_name}' from {path}")
        if id_col not in df.columns or score_col_name not in df.columns:
            logging.error(f"CSV {path} missing '{id_col}' or '{score_col_name}'. Found columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=[id_col, score_col_name])
        df[id_col] = df[id_col].astype(str).str.strip()
        return df[[id_col, score_col_name]]
    except Exception as e:
        logging.error(f"Error loading score CSV {path}: {e}")
        return pd.DataFrame(columns=[id_col, score_col_name])

def load_entropy_scores_from_pkl(run_dir_path):
    """
    Loads uncertainty scores and automated correctness labels from a run directory's PKL files.

    The function loads the uncertainty measures from the "uncertainty_measures.pkl" file and the generations data
    from the "validation_generations.pkl" file. It uses the task IDs from the uncertainty data to determine the
    order of the samples, and will use the task IDs from the generations data if they are missing from the
    uncertainty data.

    If the uncertainty data is missing, the function will log an error and return None.

    The function returns a DataFrame with columns 'id', 'semantic_entropy', 'normalized_semantic_entropy',
    'naive_entropy', and 'automated_is_correct'. The 'id' column contains the task IDs, and the other columns contain
    the respective uncertainty scores. The 'automated_is_correct' column contains the automated correctness labels
    from the uncertainty data.

    If there is an error loading the data, the function will log an error and return None.
    """
    
    run_dir = Path(run_dir_path)
    uncertainty_path = run_dir / "uncertainty_measures.pkl"
    generations_path = run_dir / "validation_generations.pkl" 
    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)
    
    if uncertainty_data is None :
        logging.error(f"Uncertainty PKL missing at {uncertainty_path} for entropy scores.")
        return None
    
    logging.info("Loading entropy scores and automated correctness labels...")
    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        
        if "task_ids" in uncertainty_data and isinstance(uncertainty_data["task_ids"], list):
            task_ids_ordered = [str(tid).strip() for tid in uncertainty_data["task_ids"]]
        elif generations_data:
            task_ids_ordered = [str(tid).strip() for tid in generations_data.keys()]
            logging.warning("Using task_ids order from generations_data as 'task_ids' key not found in uncertainty_data for entropy loading.")
        else:
            num_samples_fallback = len(measures.get("semantic_entropy", []))
            if num_samples_fallback == 0:
                logging.error("Cannot determine task order or number of samples for entropy scores.")
                return None
            task_ids_ordered = [f"sample_{i}" for i in range(num_samples_fallback)]
            logging.error("Neither uncertainty_data nor generations_data provided task_ids for entropy. Using generated sequential IDs. Alignment may be incorrect.")

        num_tasks = len(task_ids_ordered)

        def pad_or_truncate(data_list, name, target_len, default_val=np.nan):
            if data_list is None:
                logging.warning(f"Data missing for '{name}' in pkl. Will be filled with NaN.")
                return [default_val] * target_len
            current_len = len(data_list)
            if current_len < target_len:
                logging.warning(f"Padding {name}, expected {target_len} got {current_len}.")
                return list(data_list) + [default_val] * (target_len - current_len)
            elif current_len > target_len:
                logging.warning(f"Truncating {name}, expected {target_len} got {current_len}.")
                return list(data_list)[:target_len]
            return list(data_list)

        se_scores = pad_or_truncate(measures.get("semantic_entropy"), 'semantic_entropy', num_tasks)
        nse_scores = pad_or_truncate(measures.get("normalized_semantic_entropy"), 'normalized_semantic_entropy', num_tasks)
        naive_entropy = pad_or_truncate(measures.get("regular_entropy"), 'regular_entropy', num_tasks)
        validation_is_false = pad_or_truncate(uncertainty_data.get("validation_is_false"), 'validation_is_false', num_tasks, default_val=True)

        data_dict = {
            "id": task_ids_ordered,
            "semantic_entropy": se_scores,
            "normalized_semantic_entropy": nse_scores,
            "naive_entropy": naive_entropy,
            "automated_is_correct": [np.nan if pd.isna(x) else not x for x in validation_is_false]
        }
        scores_df = pd.DataFrame(data_dict)
        logging.info(f"Loaded entropy/AutoCorrect scores for {len(scores_df)} tasks.")
        return scores_df
    except Exception as e:
        logging.error(f"Error loading entropies from pkl: {e}", exc_info=True)
        return None

def load_generation_details(pkl_path, accuracy_threshold=0.5):
    """
    Loads generation details from a pickle file containing generation data.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file containing generation data.
    accuracy_threshold : float, optional
        Accuracy threshold above which a generation is considered correct for
        automated correctness labels. Defaults to 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded generation details, including
        'question', 'context', 'reference_answers', 'generated_answer',
        'accuracy_metric_score', 'is_correct_auto_from_gen', and 'response_length'.
    """
    generations_data = load_pickle(pkl_path)
    if generations_data is None: return None
    logging.info("Loading generation details...")
    data_list = []
    for task_id, task_details in generations_data.items():
        most_likely = task_details.get("most_likely_answer", {})
        reference = task_details.get("reference", {}) 
        answers = reference.get('answers', {}) 
        ref_texts = answers.get('text', [])
        
        accuracy = most_likely.get("accuracy")
        is_correct_auto = accuracy > accuracy_threshold if accuracy is not None else None
        
        resp_len_field = most_likely.get("response_length")
        if resp_len_field is None:
            response_str = most_likely.get("response", "")
            resp_len_field = len(response_str) if response_str != "[GENERATION FAILED]" else 0

        data_list.append({
            "id": str(task_id).strip(),
            "question": task_details.get("question", ""),
            "context": task_details.get("context", None),
            "reference_answers": str(ref_texts), 
            "generated_answer": most_likely.get("response", "[GENERATION FAILED]"),
            "accuracy_metric_score": most_likely.get("accuracy", np.nan),
            "is_correct_auto_from_gen": is_correct_auto,
            "response_length": resp_len_field 
        })
    df = pd.DataFrame(data_list)
    logging.info(f"Loaded generation details for {len(df)} tasks, including 'response_length'.")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge reviewed annotations with ALL scores for SUBTYPE analysis.")
    parser.add_argument("reviewed_annotations_csv", help="Path to reviewed automated annotations CSV.")
    parser.add_argument("run_files_dir", help="Path to run files directory containing uncertainty/generation pkls.")
    parser.add_argument("--is_scores_all_csv", required=True, help="Path to CSV with IS scores for ALL samples.")
    parser.add_argument("--hybrid_meta_scores_all_csv", required=True, help="Path to CSV with Hybrid Meta scores for ALL samples.")
    parser.add_argument("final_output_csv", help="Path to save final merged CSV for subtype analysis.")
    args = parser.parse_args()

    df_annotations = load_reviewed_annotations(args.reviewed_annotations_csv)
    df_entropies_auto = load_entropy_scores_from_pkl(args.run_files_dir)
    df_is_scores = load_score_csv(args.is_scores_all_csv, score_col_name='internal_signal_score')
    df_hybrid_meta = load_score_csv(args.hybrid_meta_scores_all_csv, score_col_name='hybrid_meta_score')
    gen_pkl_filename = getattr(args, 'generations_pkl_filename_override', "validation_generations.pkl")
    gen_pkl_path = Path(args.run_files_dir) / gen_pkl_filename
    df_generation_details = load_generation_details(gen_pkl_path)

    essential_dfs = {'annotations': df_annotations, 'entropies': df_entropies_auto, 'gen_details': df_generation_details}
    if any(df is None for df in essential_dfs.values()) or \
       (df_is_scores.empty and Path(args.is_scores_all_csv).exists()) or \
       (df_hybrid_meta.empty and Path(args.hybrid_meta_scores_all_csv).exists()):
        logging.error(f"One or more essential input files failed to load. Status: { {k: 'Loaded' if v is not None and not v.empty else 'FAILED/EMPTY' for k,v in essential_dfs.items()} }")
        exit(1)

    final_df = df_annotations.copy()
    logging.info(f"Starting merge with {len(final_df)} annotated samples.")

    if df_entropies_auto is not None:
        final_df = pd.merge(final_df, df_entropies_auto, on="id", how="left")
    if not df_is_scores.empty:
        final_df = pd.merge(final_df, df_is_scores, on="id", how="left")
    if not df_hybrid_meta.empty:
        final_df = pd.merge(final_df, df_hybrid_meta, on="id", how="left")
    
    cols_from_gen_details_to_merge = ['id']
    for col in ['question', 'context', 'reference_answers', 'generated_answer', 'accuracy_metric_score', 'is_correct_auto_from_gen', 'response_length']:
        if col in df_generation_details.columns and col not in final_df.columns:
            cols_from_gen_details_to_merge.append(col)
    if 'response_length' in df_generation_details.columns and 'response_length' not in cols_from_gen_details_to_merge:
        cols_from_gen_details_to_merge.append('response_length')
    
    final_df = pd.merge(final_df, df_generation_details[list(set(cols_from_gen_details_to_merge))], on="id", how="left")
    
    logging.info(f"Annotated subset row count after merging: {len(final_df)}")
    logging.info(f"Columns in final_df after all merges: {final_df.columns.tolist()}")

    se_col, is_col = 'semantic_entropy', 'internal_signal_score'
    hybrid_simple_col = 'hybrid_simple_score'
    final_df[hybrid_simple_col] = np.nan 

    if se_col in final_df.columns and is_col in final_df.columns:
        logging.info("Calculating simple hybrid score for the annotated subset...")
        scaler_se, scaler_is = MinMaxScaler(), MinMaxScaler()
        se_subset = final_df[se_col].dropna()
        is_subset = final_df[is_col].dropna()
        
        se_norm_col = f"{se_col}_norm_subset"
        is_norm_col = f"{is_col}_norm_subset"
        final_df[se_norm_col] = np.nan
        final_df[is_norm_col] = np.nan

        if not se_subset.empty:
            try:
                final_df.loc[se_subset.index, se_norm_col] = scaler_se.fit_transform(se_subset.values.reshape(-1, 1)).flatten()
            except ValueError:
                logging.warning("SE constant in subset. Normalizing to 0.5 or check data.")
                final_df.loc[se_subset.index, se_norm_col] = 0.5
        if not is_subset.empty:
            try:
                final_df.loc[is_subset.index, is_norm_col] = scaler_is.fit_transform(is_subset.values.reshape(-1, 1)).flatten()
            except ValueError:
                logging.warning("IS constant in subset. Normalizing to 0.5 or check data.")
                final_df.loc[is_subset.index, is_norm_col] = 0.5
        
        final_df[hybrid_simple_col] = 0.5 * final_df[se_norm_col] + 0.5 * final_df[is_norm_col]
        logging.info(f"Calculated '{hybrid_simple_col}'. Missing: {final_df[hybrid_simple_col].isnull().sum()}")
    else: 
        logging.warning(f"'{se_col}' or '{is_col}' not in final_df. Cannot calculate '{hybrid_simple_col}'.")

    logging.info("Final DataFrame info for subtype analysis (before final column selection):")
    final_df.info() 

    final_cols_to_select = [
        'id', 'question', 'context', 'reference_answers', 'generated_answer',
        'accuracy_metric_score', 'automated_is_correct', 'is_correct_auto_from_gen',
        'hallucination_type', 'hallucination_subtype', 'annotation_rationale',
        'semantic_entropy', 'normalized_semantic_entropy', 'naive_entropy',
        'internal_signal_score', 
        hybrid_simple_col, 
        'hybrid_meta_score',
        'response_length' 
    ]
    
    final_cols_present = [col for col in final_cols_to_select if col in final_df.columns]
    
    for critical_col in ['semantic_entropy', 'internal_signal_score', 'response_length']:
        if critical_col not in final_cols_present:
            logging.error(f"CRITICAL XAI INPUT FEATURE '{critical_col}' IS MISSING from the final selected columns. Explain_meta_learner will likely fail.")

    final_df_output = final_df[final_cols_present]

    final_path = Path(args.final_output_csv)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_df_output.to_csv(final_path, index=False)
        logging.info(f"Final subtype analysis data ({len(final_df_output)} samples, {len(final_df_output.columns)} cols) saved to: {final_path}")
        logging.info(f"Columns saved: {final_df_output.columns.tolist()}")
    except IOError as e:
        logging.error(f"Error writing final file {final_path}: {e}")
    
    logging.info("\n--- Data Preparation Complete ---")