import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from src.utils.logging_utils import setup_logger
from src.utils.utils import load_pickle

def train_and_predict_meta(run_id, run_dir):
    """
    Train and predict a hybrid meta-learner to combine internal signal scores and semantic uncertainty measures.

    Parameters
    ----------
    run_id : str
        The run ID for the experiment.
    run_dir : str
        The path to the run directory containing the uncertainty_measures.pkl and validation_generations.pkl files.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    setup_logger(); run_dir = Path(run_dir)
    logging.info(f"--- Training & Predicting Hybrid Meta-Learner for Run: {run_id} ---")

    uncertainty_path = run_dir / "uncertainty_measures.pkl"
    generations_path = run_dir / "validation_generations.pkl"
    is_scores_path = run_dir / f"{run_id}_internal_signal_scores_all.csv"

    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)
    if not is_scores_path.is_file(): logging.error(f"IS scores file not found: {is_scores_path}"); return False
    df_is = pd.read_csv(is_scores_path); df_is['id'] = df_is['id'].astype(str).str.strip()

    if uncertainty_data is None or generations_data is None: return False
    if 'internal_signal_score' not in df_is.columns: logging.error("IS scores CSV missing column."); return False

    try:
        measures = uncertainty_data.get("uncertainty_measures", {})
        se_scores = measures.get("semantic_entropy")
        validation_is_false = uncertainty_data.get("validation_is_false")
        task_ids_ordered = list(generations_data.keys()); num_tasks = len(task_ids_ordered)
        if se_scores is None or validation_is_false is None or len(se_scores) != num_tasks or len(validation_is_false) != num_tasks:
            logging.error("SE scores or validation_is_false missing/mismatch."); return False
        df_se = pd.DataFrame({'id': [str(tid).strip() for tid in task_ids_ordered], 'semantic_entropy': se_scores, 'is_incorrect_auto': validation_is_false})
    except Exception as e: logging.error(f"Error processing PKL: {e}"); return False

    merged_df = pd.merge(df_se, df_is[['id', 'internal_signal_score']], on='id', how='inner')
    merged_df.dropna(subset=['semantic_entropy', 'internal_signal_score', 'is_incorrect_auto'], inplace=True)
    logging.info(f"Using {len(merged_df)} complete samples for meta-learner training.")
    if len(merged_df) < 20 or len(merged_df['is_incorrect_auto'].unique()) < 2: logging.error("Insufficient data/classes."); return False

    scaler_se = MinMaxScaler(); scaler_is = MinMaxScaler()
    merged_df['se_norm'] = scaler_se.fit_transform(merged_df[['semantic_entropy']])
    merged_df['is_norm'] = scaler_is.fit_transform(merged_df[['internal_signal_score']])
    scalers_path = run_dir / f"{run_id}_hybrid_meta_scalers.pkl"
    try:
        with open(scalers_path, 'wb') as f: pickle.dump({'scaler_se': scaler_se, 'scaler_is': scaler_is}, f)
        logging.info(f"Saved fitted scalers to: {scalers_path}")
    except Exception as e: logging.error(f"Error saving scalers: {e}")

    X_full = merged_df[['se_norm', 'is_norm']]
    y_full = merged_df['is_incorrect_auto']
    meta_classifier = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    logging.info("Training hybrid meta-learner on full valid data...")
    try: meta_classifier.fit(X_full, y_full)
    except Exception as e: logging.error(f"Error training meta-learner: {e}"); return False

   
    model_path = run_dir / f"{run_id}_hybrid_meta_model.pkl"
    try:
        with open(model_path, 'wb') as f: pickle.dump(meta_classifier, f)
        logging.info(f"Saved trained hybrid meta-learner to: {model_path}")
    except Exception as e: logging.error(f"Error saving meta model: {e}"); return False

    logging.info("Predicting Hybrid Meta scores for ALL validation samples...")
    df_se_all = pd.DataFrame({'id': [str(tid).strip() for tid in task_ids_ordered], 'semantic_entropy': se_scores})
    df_is_all = pd.read_csv(is_scores_path); df_is_all['id'] = df_is_all['id'].astype(str).str.strip()
    merged_all_df = pd.merge(df_se_all, df_is_all, on='id', how='left')

    predict_mask_all = merged_all_df[['semantic_entropy', 'internal_signal_score']].notna().all(axis=1)
    valid_all_df = merged_all_df[predict_mask_all].copy(); invalid_all_df = merged_all_df[~predict_mask_all].copy()

    if not valid_all_df.empty:
        try:
            valid_all_df['se_norm'] = scaler_se.transform(valid_all_df[['semantic_entropy']])
            valid_all_df['is_norm'] = scaler_is.transform(valid_all_df[['internal_signal_score']])
        except Exception as e: logging.error(f"Error applying saved scalers: {e}"); valid_all_df['hybrid_meta_score'] = np.nan
        else:
            X_predict_all = valid_all_df[['se_norm', 'is_norm']]
            try:
                probabilities_all = meta_classifier.predict_proba(X_predict_all)
                true_class_idx = np.where(meta_classifier.classes_ == True)[0][0]
                valid_all_df['hybrid_meta_score'] = probabilities_all[:, true_class_idx]
            except Exception as e: logging.error(f"Error predicting full set hybrid scores: {e}"); valid_all_df['hybrid_meta_score'] = np.nan

    invalid_all_df['hybrid_meta_score'] = np.nan
    results_df_all = pd.concat([valid_all_df, invalid_all_df], ignore_index=True)[['id', 'hybrid_meta_score']]

    output_path_all = run_dir / f"{run_id}_hybrid_meta_scores_all.csv"
    try: results_df_all.to_csv(output_path_all, index=False); logging.info(f"Saved full hybrid meta scores ({len(results_df_all)} rows) to: {output_path_all}")
    except IOError as e: logging.error(f"Error writing hybrid meta scores file: {e}"); return False

    logging.info(f"--- Hybrid Meta-Learner Training & Prediction Complete: {run_id} ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid meta-learner & predict scores.")
    parser.add_argument("run_id", type=str); parser.add_argument("run_dir", type=str)
    args = parser.parse_args(); train_and_predict_meta(args.run_id, args.run_dir)