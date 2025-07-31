import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import shap 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression

from src.utils.logging_utils import setup_logger

SCALED_FEATURE_NAMES = ['Semantic Entropy (Norm.)', 'Internal Signal Score (Norm.)', 'Response Length (Norm.)']

def explain_hybrid_meta_learner(
    run_id_arg: str, 
    run_dir_str: str, 
    meta_model_filename: str,
    final_analysis_data_csv_filename: str, 
    scalers_filename: str, 
    num_shap_summary_samples: int = 100,
    num_instance_plots: int = 3,
    specific_instance_ids_to_explain: list = None 
):
    """
    Explain the hybrid meta-learner using SHAP values.

    This function takes a directory with a meta-learner model, scalers, and analysis data,
    and generates SHAP summary plots and instance-level SHAP explanations.

    Parameters
    ----------
    run_id_arg : str
        The ID of the run that produced the data in the provided directory.
    run_dir_str : str
        The path to the directory containing the meta-learner model, scalers, and analysis data.
    meta_model_filename : str
        The name of the file containing the saved meta-learner model.
    final_analysis_data_csv_filename : str
        The name of the file containing the final analysis data.
    scalers_filename : str
        The name of the file containing the saved scalers.
    num_shap_summary_samples : int
        The number of samples to use for the SHAP summary plot.
    num_instance_plots : int
        The number of instances to generate SHAP waterfall plots for.
    specific_instance_ids_to_explain : list
        A list of instance IDs to generate SHAP waterfall plots for. If empty, will randomly
        select instances from the analysis data.

    Returns
    -------
    importance_df : pd.DataFrame
        A DataFrame containing feature importance values for the hybrid meta-learner.
    """
    setup_logger()
    run_dir = Path(run_dir_str)
    logging.info(f"--- Explaining Hybrid Meta-Learner for Run: {run_id_arg} ---")

    model_path = run_dir / meta_model_filename
    scalers_path = run_dir / scalers_filename
    analysis_data_path = run_dir / final_analysis_data_csv_filename
    
    shap_plot_dir = run_dir / "xai_hybrid_meta_plots"
    shap_plot_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        logging.error(f"Meta-learner model not found: {model_path}. Cannot explain.")
        return None
    try:
        with open(model_path, 'rb') as f: meta_learner = pickle.load(f)
        logging.info(f"Loaded meta-learner ({type(meta_learner).__name__}) from: {model_path}")
    except Exception as e:
        logging.error(f"Error loading meta-learner model: {e}")
        return None

    if not scalers_path.is_file():
        logging.error(f"Scalers file not found: {scalers_path}. Cannot perform SHAP/explanation accurately."); return None
    try:
        with open(scalers_path, 'rb') as f: final_scalers = pickle.load(f)
        logging.info(f"Loaded scalers from: {scalers_path}")
        
        num_expected_features_for_model = 0
        if hasattr(meta_learner, 'n_features_in_'):
            num_expected_features_for_model = meta_learner.n_features_in_
        elif hasattr(meta_learner, 'coef_'):
            num_expected_features_for_model = meta_learner.coef_.shape[-1]
        elif hasattr(meta_learner, 'feature_importances_'):
            num_expected_features_for_model = len(meta_learner.feature_importances_)

        if not (isinstance(final_scalers, list) and len(final_scalers) == num_expected_features_for_model):
            logging.error(f"Loaded scalers list length ({len(final_scalers)}) does not match "
                          f"model's expected number of features ({num_expected_features_for_model}). "
                          f"Ensure SCALED_FEATURE_NAMES matches the features meta_learner was trained on ({len(SCALED_FEATURE_NAMES)} names defined)."); 
            if len(SCALED_FEATURE_NAMES) != num_expected_features_for_model :
                 logging.warning(f"SCALED_FEATURE_NAMES length is {len(SCALED_FEATURE_NAMES)} but model expects {num_expected_features_for_model}")
            return None
    except Exception as e:
        logging.error(f"Error loading scalers: {e}")
        return None
        
    if not analysis_data_path.is_file():
        logging.error(f"Final analysis data CSV not found: {analysis_data_path}. Cannot get data for SHAP.")
        return None
    try:
        df_full_data = pd.read_csv(analysis_data_path)
        df_full_data['id'] = df_full_data['id'].astype(str) 
        
        current_scaled_feature_names = SCALED_FEATURE_NAMES[:len(final_scalers)]
        raw_feature_cols_map = {
            'Semantic Entropy (Norm.)': 'semantic_entropy',
            'Internal Signal Score (Norm.)': 'internal_signal_score',
            'Response Length (Norm.)': 'response_length'
        }
        current_raw_feature_cols = [raw_feature_cols_map[sf_name] for sf_name in current_scaled_feature_names if sf_name in raw_feature_cols_map]

        if not all(col in df_full_data.columns for col in current_raw_feature_cols):
            missing_cols = [col for col in current_raw_feature_cols if col not in df_full_data.columns]
            logging.error(f"CSV {analysis_data_path} missing required raw feature columns: {missing_cols}. Available: {df_full_data.columns.tolist()}")
            return None
        
        df_features_raw = df_full_data.set_index('id')[current_raw_feature_cols].copy().dropna()
        if df_features_raw.empty:
            logging.error("No valid data rows after dropna on raw features for SHAP.")
            return None

        X_scaled_for_shap = np.zeros((df_features_raw.shape[0], len(current_raw_feature_cols)), dtype=float)
        for i, col_name_raw in enumerate(current_raw_feature_cols):
            scaler = final_scalers[i] 
            raw_values = df_features_raw[col_name_raw].values.reshape(-1, 1)
            try:
                X_scaled_for_shap[:, i] = scaler.transform(raw_values).flatten()
            except ValueError as ve: 
                logging.warning(f"SHAP scaling: ValueError for {col_name_raw}: {ve}. Setting to 0.5.")
                X_scaled_for_shap[:, i] = 0.5 
            except Exception as e_scale:
                logging.error(f"Error scaling feature {col_name_raw} for SHAP: {e_scale}")
                return None

        X_scaled_shap_df = pd.DataFrame(X_scaled_for_shap, columns=current_scaled_feature_names, index=df_features_raw.index)
        logging.info(f"Prepared {X_scaled_shap_df.shape[0]} samples with {X_scaled_shap_df.shape[1]} scaled features for SHAP analysis.")
    except Exception as e:
        logging.error(f"Error loading or preparing data for SHAP: {e}", exc_info=True)
        return None

    importance_df = None
    if isinstance(meta_learner, LogisticRegression) and hasattr(meta_learner, 'coef_'):
        coefficients = meta_learner.coef_[0]
        if len(coefficients) == X_scaled_shap_df.shape[1]:
            importance_df = pd.DataFrame({'Feature': X_scaled_shap_df.columns, 'Coefficient': coefficients})
            importance_df = importance_df.sort_values(by='Coefficient', key=abs, ascending=False)
            logging.info("Hybrid Meta-Learner (Logistic Regression) Coefficients (for P(Incorrect)):")
            logging.info(importance_df.to_string())
        else:
            logging.warning("LR coefficients length mismatch with features.")
    elif hasattr(meta_learner, 'feature_importances_'): 
        importances = meta_learner.feature_importances_
        if len(importances) == X_scaled_shap_df.shape[1]:
            importance_df = pd.DataFrame({'Feature': X_scaled_shap_df.columns, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            logging.info(f"Hybrid Meta-Learner ({type(meta_learner).__name__}) Feature Importances:")
            logging.info(importance_df.to_string())
        else:
            logging.warning("Feature importances length mismatch with features.")
    else:
        logging.warning(f"Meta-learner type {type(meta_learner).__name__} does not have standard 'coef_' or 'feature_importances_'.")
    
    try:
        explainer = None
        if isinstance(meta_learner, RandomForestClassifier):
            logging.info("Initializing SHAP TreeExplainer for RandomForest.")
            explainer = shap.TreeExplainer(meta_learner)
        elif isinstance(meta_learner, LogisticRegression):
            logging.info("Initializing SHAP LinearExplainer for LogisticRegression.")
            explainer = shap.LinearExplainer(meta_learner, X_scaled_shap_df)
        else:
            logging.warning(f"No specific SHAP explainer for {type(meta_learner).__name__}. Attempting KernelExplainer.")
            if X_scaled_shap_df.shape[0] > 1:
                 background_data = shap.sample(X_scaled_shap_df, min(50, X_scaled_shap_df.shape[0]), random_state=42)
                 explainer = shap.KernelExplainer(meta_learner.predict_proba, background_data)
            else: logging.error("Not enough data for KernelExplainer background.")

        if explainer is None:
            logging.error("Could not initialize SHAP explainer."); return importance_df

        X_for_summary_plot = X_scaled_shap_df
        if X_scaled_shap_df.shape[0] > num_shap_summary_samples:
            X_for_summary_plot = X_scaled_shap_df.sample(n=num_shap_summary_samples, random_state=42)
        
        shap_values_raw_for_summary = explainer.shap_values(X_for_summary_plot) 

        positive_class_index = 1 
        n_classes_model = getattr(meta_learner, 'n_classes_', 2)
        if hasattr(meta_learner, 'classes_'):
            classes_list = list(meta_learner.classes_)
            if True in classes_list: positive_class_index = classes_list.index(True)
            elif 1 in classes_list: positive_class_index = classes_list.index(1)
        logging.info(f"Meta-learner classes: {getattr(meta_learner, 'classes_', 'N/A')}. Using index {positive_class_index} for P(Incorrect) SHAP values.")
        
        shap_values_for_positive_class_summary = None
        if isinstance(shap_values_raw_for_summary, list) and len(shap_values_raw_for_summary) == n_classes_model:
            shap_values_for_positive_class_summary = shap_values_raw_for_summary[positive_class_index]
        elif isinstance(explainer, shap.explainers.Linear) and isinstance(shap_values_raw_for_summary, np.ndarray) and shap_values_raw_for_summary.ndim == 2:
            shap_values_for_positive_class_summary = shap_values_raw_for_summary
        elif isinstance(shap_values_raw_for_summary, np.ndarray) and shap_values_raw_for_summary.ndim == 3 and shap_values_raw_for_summary.shape[-1] == n_classes_model:
            shap_values_for_positive_class_summary = shap_values_raw_for_summary[:, :, positive_class_index]
        elif isinstance(shap_values_raw_for_summary, np.ndarray) and shap_values_raw_for_summary.ndim == 2 and n_classes_model == 2:
             shap_values_for_positive_class_summary = shap_values_raw_for_summary 
        else:
            logging.error(f"Unexpected SHAP values format for summary: type {type(shap_values_raw_for_summary)}, shape if array: {getattr(shap_values_raw_for_summary, 'shape', 'N/A')}")
            return importance_df

        if shap_values_for_positive_class_summary is None or shap_values_for_positive_class_summary.shape[0] != X_for_summary_plot.shape[0] or shap_values_for_positive_class_summary.shape[1] != X_for_summary_plot.shape[1]:
             logging.error(f"Final SHAP values for positive class summary shape is problematic. SHAP values shape: {getattr(shap_values_for_positive_class_summary, 'shape', 'None')}, Feature sample shape: {X_for_summary_plot.shape}.")
        else:
            logging.info("Generating SHAP summary plot...")
            plt.figure()
            shap.summary_plot(shap_values_for_positive_class_summary, X_for_summary_plot, plot_type="bar", show=False)
            plt.title(f"SHAP Mean Abs. Feat. Contrib. to P(Incorrect) ({type(meta_learner).__name__}, {run_id_arg})")
            plt.savefig(shap_plot_dir / f"{run_id_arg}_shap_summary_bar.png", bbox_inches='tight')
            plt.close()
            logging.info(f"Saved SHAP summary bar plot to {shap_plot_dir}")
        
        instances_to_explain_df = pd.DataFrame()
        ids_for_waterfall = []

        if specific_instance_ids_to_explain:
            ids_for_waterfall = [str(sid) for sid in specific_instance_ids_to_explain if str(sid) in X_scaled_shap_df.index]
            if len(ids_for_waterfall) < len(specific_instance_ids_to_explain):
                logging.warning(f"Not all specified instance IDs found. Found {len(ids_for_waterfall)} of {len(specific_instance_ids_to_explain)}.")
            if not ids_for_waterfall and num_instance_plots > 0:
                logging.warning("Specified IDs not found, will pick random ones if num_instance_plots > 0.")
            elif not ids_for_waterfall:
                logging.warning("No specified instance IDs found and num_instance_plots is 0 or less.")
        
        if not ids_for_waterfall and num_instance_plots > 0 and not X_scaled_shap_df.empty:
            num_to_sample_randomly = min(num_instance_plots, X_scaled_shap_df.shape[0])
            ids_for_waterfall = X_scaled_shap_df.sample(n=num_to_sample_randomly, random_state=42).index.tolist()
            logging.info(f"Plotting {len(ids_for_waterfall)} randomly selected instances for SHAP waterfall.")
        
        if ids_for_waterfall:
            instances_to_explain_df = X_scaled_shap_df.loc[ids_for_waterfall]
            logging.info(f"Plotting {instances_to_explain_df.shape[0]} instances for SHAP waterfall.")

            shap_values_instances_raw = explainer.shap_values(instances_to_explain_df)
            shap_values_for_instances_positive_class = None

            if isinstance(shap_values_instances_raw, list) and len(shap_values_instances_raw) == n_classes_model:
                shap_values_for_instances_positive_class = shap_values_instances_raw[positive_class_index]
            elif isinstance(explainer, shap.explainers.Linear) and isinstance(shap_values_instances_raw, np.ndarray) and shap_values_instances_raw.ndim == 2:
                shap_values_for_instances_positive_class = shap_values_instances_raw
            elif isinstance(shap_values_instances_raw, np.ndarray) and shap_values_instances_raw.ndim == 3 and shap_values_instances_raw.shape[-1] == n_classes_model:
                shap_values_for_instances_positive_class = shap_values_instances_raw[:, :, positive_class_index]
            elif isinstance(shap_values_instances_raw, np.ndarray) and shap_values_instances_raw.ndim == 2 and n_classes_model == 2:
                 shap_values_for_instances_positive_class = shap_values_instances_raw
            else:
                logging.error(f"Unexpected SHAP values format for instances: type {type(shap_values_instances_raw)}")
                return importance_df
            
            if shap_values_for_instances_positive_class is None or shap_values_for_instances_positive_class.shape[0] != instances_to_explain_df.shape[0] or shap_values_for_instances_positive_class.shape[1] != instances_to_explain_df.shape[1]:
                logging.error(f"SHAP values for instances positive class have unexpected shape or are None. SHAP Shape: {getattr(shap_values_for_instances_positive_class, 'shape', 'None')}. Expected Features DF Shape: {instances_to_explain_df.shape}")
                return importance_df

            base_value_for_plot = explainer.expected_value
            if isinstance(base_value_for_plot, (np.ndarray, list)) and len(base_value_for_plot) == n_classes_model: 
                 base_value_for_plot = explainer.expected_value[positive_class_index]
            elif isinstance(base_value_for_plot, (np.ndarray, list)) and len(base_value_for_plot) == 1:
                 base_value_for_plot = explainer.expected_value[0]
            elif not isinstance(base_value_for_plot, (float, np.floating, int, np.integer)):
                 logging.error(f"explainer.expected_value is not suitable. Type: {type(base_value_for_plot)}, Value: {base_value_for_plot}")
                 return importance_df

            logging.info("\n--- Instance-Level SHAP Explanations ---")
            for i in range(instances_to_explain_df.shape[0]):
                instance_id_from_index = str(instances_to_explain_df.index[i]) 
                current_shap_values_for_instance = shap_values_for_instances_positive_class[i]
                current_data_values = instances_to_explain_df.iloc[i].values
                
                if not (current_shap_values_for_instance.ndim == 1 and len(current_shap_values_for_instance) == instances_to_explain_df.shape[1]):
                    logging.error(f"SHAP values for instance {instance_id_from_index} have unexpected shape: {current_shap_values_for_instance.shape}. Skipping.")
                    continue
                
                original_instance_data_series = df_full_data[df_full_data['id'] == instance_id_from_index]
                q_text, a_text, subtype_text, ref_text = "N/A", "N/A", "N/A", "N/A"
                if not original_instance_data_series.empty:
                    original_instance_data = original_instance_data_series.iloc[0]
                    q_text = original_instance_data.get('question', 'N/A')0
                    a_text = original_instance_data.get('generated_answer', 'N/A')
                    subtype_text = original_instance_data.get('hallucination_subtype', 'N/A')

                    ref_text = original_instance_data.get('reference_answers', 'N/A')
                else:
                    logging.warning(f"Could not find original data for ID {instance_id_from_index} in df_full_data for logging.")

                logging.info(f"\nInstance ID: {instance_id_from_index}")
                logging.info(f"  Question: {q_text}")
                logging.info(f"  LLM Answer: {a_text}")
                logging.info(f"  Reference Answer(s): {ref_text}")
                logging.info(f"  Annotated Subtype: {subtype_text}")
                logging.info(f"  Scaled Features for SHAP: {dict(zip(instances_to_explain_df.columns, current_data_values))}")
                logging.info(f"  SHAP Values (P(Incorrect)): {dict(zip(instances_to_explain_df.columns, current_shap_values_for_instance))}")
                
                predicted_proba_instance = meta_learner.predict_proba(current_data_values.reshape(1, -1))[0]
                logging.info(f"  Base P(Incorrect): {base_value_for_plot:.4f}, Predicted P(Class0): {predicted_proba_instance[0]:.4f}, Predicted P(Class1/Incorrect): {predicted_proba_instance[positive_class_index]:.4f}")

                plot_title = (f"SHAP Waterfall P(Incorrect) - ID: {instance_id_from_index} ({run_id_arg})")
                
                plt.figure(figsize=(10,4)) 
                try:
                    shap.waterfall_plot(shap.Explanation(values=current_shap_values_for_instance,
                                                         base_values=base_value_for_plot, 
                                                         data=current_data_values,
                                                         feature_names=instances_to_explain_df.columns.tolist()), 
                                         show=False, max_display=len(current_scaled_feature_names) + 2) 
                    plt.title(plot_title, fontsize=12) 
                    plt.tight_layout()
                    plt.savefig(shap_plot_dir / f"{run_id_arg}_shap_waterfall_id_{instance_id_from_index}.png", bbox_inches='tight')
                except ValueError as ve: logging.error(f"Plotting VE for {instance_id_from_index}: {ve}")
                except Exception as e_plot: logging.error(f"Plotting E for {instance_id_from_index}: {e_plot}")
                finally: plt.close()
            logging.info(f"Attempted to save {instances_to_explain_df.shape[0]} SHAP waterfall plots.")
        else:
            logging.info("No instances selected for SHAP waterfall plots based on criteria.")
    except ImportError:
        logging.error("SHAP library not installed. Please install it (`pip install shap`) to generate SHAP explanations.")
    except Exception as e:
        logging.error(f"Error during SHAP explanation: {e}", exc_info=True)
        
    return importance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain Hybrid Meta-Learner.")
    parser.add_argument("run_id", type=str, help="Run ID for naming outputs.")
    parser.add_argument("run_dir", type=str, help="Directory containing run files.")
    
    parser.add_argument("--meta_model_filename", default=None)
    parser.add_argument("--final_analysis_data_csv", default=None)
    parser.add_argument("--scalers_filename", default=None)

    parser.add_argument("--num_shap_summary_samples", type=int, default=100)
    parser.add_argument("--num_instance_plots", type=int, default=3)
    parser.add_argument("--specific_instance_ids", nargs='*', default=None)

    args = parser.parse_args()
    
    meta_model_fname_default = f"{args.run_id}_hybrid_meta_model.pkl"
    analysis_csv_default = f"{args.run_id}_final_analysis_data.csv" 
    scalers_fname_default = f"{args.run_id}_hybrid_meta_scalers.pkl"

    explain_hybrid_meta_learner(
        args.run_id,
        args.run_dir,
        meta_model_filename=args.meta_model_filename if args.meta_model_filename else meta_model_fname_default,
        final_analysis_data_csv_filename=args.final_analysis_data_csv if args.final_analysis_data_csv else analysis_csv_default,
        scalers_filename=args.scalers_filename if args.scalers_filename else scalers_fname_default,
        num_shap_summary_samples=args.num_shap_summary_samples,
        num_instance_plots=args.num_instance_plots,
        specific_instance_ids_to_explain=args.specific_instance_ids
    )