import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import logging
from pathlib import Path
from sklearn.metrics import auc
from src.utils.logging_utils import setup_logger

plt.style.use("seaborn-v0_8-darkgrid")

def save_plot(figure, filepath):
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filepath, bbox_inches="tight", dpi=150)
        logging.info(f"Plot saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filepath}: {e}")
    finally:
        plt.close(figure)

def load_and_tag_run_data(csv_path, run_id):
    try:
        df = pd.read_csv(csv_path)
        df['run_id'] = run_id
        logging.info(f"Loaded {len(df)} rows for run {run_id} from {csv_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {csv_path}: {e}")
        return None

def normalize_scores_within_runs(df, score_columns):
    df_normalized = df.copy()
    required_cols = ['run_id'] + score_columns
    if not all(col in df_normalized.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_normalized.columns]
        logging.error(f"DataFrame is missing required columns for normalization: {missing}")
        return df

    for col in score_columns:
        if col not in df_normalized.columns:
            logging.warning(f"Score column '{col}' not found for normalization, skipping.")
            continue
        normalized_col_name = f"{col}_z"
        means = df_normalized.groupby('run_id')[col].transform('mean')
        stds = df_normalized.groupby('run_id')[col].transform('std')
        stds = stds.replace(0, 1)
        df_normalized[normalized_col_name] = (df_normalized[col] - means) / stds
        df_normalized[normalized_col_name] = df_normalized[normalized_col_name].fillna(0)

    logging.info(f"Applied Z-score normalization to: {score_columns}")
    return df_normalized

def perform_aggregate_subtype_analysis(
    df_aggregated_normalized,
    score_column_normalized,
    dataset_name,
    min_samples_per_subtype=2,
    plot_dir=None,
    save_plots=False,
):
    score_column_original = score_column_normalized.replace('_z', '')
    logging.info(
        f"\n--- Analyzing Aggregated Normalized Score: '{score_column_normalized}' for Dataset: {dataset_name} ---"
    )

    if score_column_normalized not in df_aggregated_normalized.columns:
        logging.error(f"Normalized score column '{score_column_normalized}' not found. Skipping.")
        return
    if df_aggregated_normalized[score_column_normalized].isnull().all():
        logging.warning(f"All values in '{score_column_normalized}' are NaN. Skipping.")
        return

    df_annotated = df_aggregated_normalized.dropna(subset=['hallucination_type', 'hallucination_subtype', score_column_normalized]).copy()
    num_annotated = len(df_annotated)
    logging.info(f"Using {num_annotated} valid annotated rows for {dataset_name} aggregate analysis for score {score_column_normalized}.")
    if num_annotated == 0:
        logging.warning("No valid annotated rows remaining after filtering NaNs.")
        return

    print(f"\nBasic Stats for '{score_column_normalized}' (Aggregated, Normalized):")
    print(df_annotated[score_column_normalized].describe())

    fig_type, ax_type = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_annotated, x="hallucination_type", y=score_column_normalized, ax=ax_type, palette="viridis", hue="hallucination_type", legend=False)
    sns.stripplot(data=df_annotated, x="hallucination_type", y=score_column_normalized, color=".25", size=3, alpha=0.5, ax=ax_type)
    ax_type.set_title(f"Aggregated Z-Score ({score_column_original}) Dist. by Type ({dataset_name})")
    ax_type.set_xlabel("Primary Hallucination Type"); ax_type.set_ylabel(f"Normalized Score ({score_column_original}_z)")
    ax_type.grid(axis="y", linestyle="--", alpha=0.7)
    if save_plots and plot_dir: save_plot(fig_type, plot_dir / f"Agg_{dataset_name}_{score_column_original}_vs_Type.png")
    else: plt.close(fig_type)

    subtype_counts = df_annotated["hallucination_subtype"].value_counts()
    subtypes_to_plot = subtype_counts[subtype_counts >= min_samples_per_subtype].index
    df_plot_subtype = df_annotated[df_annotated["hallucination_subtype"].isin(subtypes_to_plot)].copy()

    if not df_plot_subtype.empty:
        fig_subtype, ax_subtype = plt.subplots(figsize=(14, 8))
        median_scores = df_plot_subtype.groupby("hallucination_subtype")[score_column_normalized].median()
        order = median_scores.sort_values().index
        sns.boxplot(data=df_plot_subtype, x="hallucination_subtype", y=score_column_normalized, order=order, ax=ax_subtype, palette="viridis", hue="hallucination_subtype", legend=False)
        ax_subtype.set_title(f"Aggregated Z-Score ({score_column_original}) Dist. by Subtype ({dataset_name}, n >= {min_samples_per_subtype})")
        ax_subtype.set_xlabel("Hallucination Subtype"); ax_subtype.set_ylabel(f"Normalized Score ({score_column_original}_z)")
        ax_subtype.tick_params(axis="x", rotation=45); ax_subtype.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        if save_plots and plot_dir: save_plot(fig_subtype, plot_dir / f"Agg_{dataset_name}_{score_column_original}_vs_Subtype.png")
        else: plt.close(fig_subtype)
    else:
        logging.warning(f"Not enough samples per subtype (min={min_samples_per_subtype}) for {score_column_normalized}. Skipping subtype plot.")

    print(f"\n--- AGGREGATE Summary Stats for {score_column_normalized} by Type ({dataset_name}) ---")
    print(df_annotated.groupby("hallucination_type")[score_column_normalized].agg(["mean", "median", "std", "count"]))
    if not df_plot_subtype.empty:
        print(f"\n--- AGGREGATE Summary Stats for {score_column_normalized} by Subtype ({dataset_name}, n >= {min_samples_per_subtype}) ---")
        print(df_plot_subtype.groupby("hallucination_subtype")[score_column_normalized].agg(["mean", "median", "std", "count"]).sort_values(by="median"))
    else:
        logging.warning(f"Skipping AGGREGATE subtype summary statistics due to insufficient data for {score_column_normalized}.")

    def run_mw_test(group1_name, group2_name, justification, data_df, score_col, group_by_col):
        print(f"\n--- AGGREGATE Mann-Whitney U test: '{group1_name}' vs '{group2_name}' ({score_col}) ---")
        if justification: print(f"--- Justification: {justification} ---")

        df_testable = data_df[data_df[group_by_col].isin([group1_name, group2_name])].copy()
        group1_exists = group1_name in df_testable[group_by_col].unique()
        group2_exists = group2_name in df_testable[group_by_col].unique()

        if group1_exists and group2_exists:
            scores_group1 = df_testable[df_testable[group_by_col] == group1_name][score_col]
            scores_group2 = df_testable[df_testable[group_by_col] == group2_name][score_col]
            n1, n2 = len(scores_group1), len(scores_group2)
            var1, var2 = np.var(scores_group1, ddof=1), np.var(scores_group2, ddof=1)

            if (n1 > 1 and n2 > 1 and not np.isclose(var1, 0) and not np.isclose(var2, 0)):
                try:
                    stat, p_value = stats.mannwhitneyu(scores_group1, scores_group2, alternative="two-sided", nan_policy='omit')
                    rank_biserial_r = 1 - (2 * stat) / (n1 * n2)
                    print(f"Comparing {n1} samples ({group1_name}) vs {n2} samples ({group2_name}).")
                    print(f"Statistic U: {stat:.3f}, P-value: {p_value:.4f}")
                    print(f"Effect Size (Rank Biserial r): {rank_biserial_r:.3f}")
                    if p_value < 0.05: print("Result: Statistically significant difference (p < 0.05)")
                    else: print("Result: No statistically significant difference (p >= 0.05)")
                except ValueError as e: print(f"Could not perform Mann-Whitney U test: {e}")
            else: print(f"Skipping test: Insufficient samples or zero variance. '{group1_name}': n={n1}, var={var1:.2E}; '{group2_name}': n={n2}, var={var2:.2E}.")
        else:
            missing = [g for g, exists in zip([group1_name, group2_name], [group1_exists, group2_exists]) if not exists]
            print(f"Skipping test: One or both groups ({', '.join(missing)}) not found in filtered data (n >= {min_samples_per_subtype}).")

    run_mw_test("Factuality", "Faithfulness", "Comparing broad primary types", df_annotated, score_column_normalized, group_by_col='hallucination_type')
    run_mw_test("A3_DiagnosticCriteriaDefinition", "B1_ExtrapolationAddition", "Comparing most frequent Factuality vs Faithfulness subtype", df_plot_subtype, score_column_normalized, group_by_col='hallucination_subtype')
    run_mw_test("A3_DiagnosticCriteriaDefinition", "A2_ContraindicationIndication", "Comparing conceptual factual errors vs. specific indication errors", df_plot_subtype, score_column_normalized, group_by_col='hallucination_subtype')
    run_mw_test("A3_DiagnosticCriteriaDefinition", "A1_ContextIgnorant", "Comparing conceptual factual errors vs. context ignorance errors", df_plot_subtype, score_column_normalized, group_by_col='hallucination_subtype')

def main():
    parser = argparse.ArgumentParser(description="Aggregate and analyze hallucination subtypes vs uncertainty scores across runs for specific datasets.")
    parser.add_argument("--base_output_dir", type=str, required=True, help="Base directory containing individual run output folders.")
    parser.add_argument("--bioasq_run_ids", nargs='+', required=True, help="List of run IDs corresponding to BioASQ experiments.")
    parser.add_argument("--medquad_run_ids", nargs='+', required=True, help="List of run IDs corresponding to MedQuAD experiments.")
    parser.add_argument("--output_dir", type=str, default="./outputs/analysis_aggregated", help="Directory to save aggregated results and plots.")
    parser.add_argument("--min_samples", type=int, default=2, help="Minimum samples required per subtype for detailed aggregate plotting and stats.")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files.")

    args = parser.parse_args()
    setup_logger()

    base_dir = Path(args.base_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_process = {
        "BioASQ": args.bioasq_run_ids,
        "MedQuAD": args.medquad_run_ids
    }

    score_cols_to_normalize = [
        'semantic_entropy', 'normalized_semantic_entropy', 'naive_entropy',
        'internal_signal_score', 'hybrid_simple_score', 'hybrid_meta_score'
    ]

    for dataset_name, run_ids in datasets_to_process.items():
        logging.info(f"\n{'='*20} Processing Aggregated Data for: {dataset_name} {'='*20}")
        all_dfs_dataset = []
        for run_id in run_ids:
            run_dir = base_dir / run_id
            csv_path = run_dir / f"{run_id}_final_analysis_data.csv"
            df_run = load_and_tag_run_data(csv_path, run_id)
            if df_run is not None:
                all_dfs_dataset.append(df_run)

        if not all_dfs_dataset:
            logging.warning(f"No data loaded for dataset {dataset_name}. Skipping.")
            continue

        df_aggregated = pd.concat(all_dfs_dataset, ignore_index=True)
        logging.info(f"Aggregated data for {dataset_name}. Total rows: {len(df_aggregated)}")

        cols_present = df_aggregated.columns.tolist()
        valid_score_cols_to_normalize = [col for col in score_cols_to_normalize if col in cols_present]
        if not valid_score_cols_to_normalize:
             logging.error(f"None of the target score columns found in aggregated data for {dataset_name}. Columns found: {cols_present}")
             continue
        logging.info(f"Found columns for normalization in {dataset_name}: {valid_score_cols_to_normalize}")

        required_analysis_cols = valid_score_cols_to_normalize + ['hallucination_type', 'hallucination_subtype', 'run_id']
        if not all(col in df_aggregated.columns for col in required_analysis_cols):
            missing = [col for col in required_analysis_cols if col not in df_aggregated.columns]
            logging.error(f"Aggregated DataFrame for {dataset_name} is missing required columns for analysis: {missing}. Check input CSVs.")
            continue

        df_normalized = normalize_scores_within_runs(df_aggregated, valid_score_cols_to_normalize)

        analysis_cols_normalized = [f"{col}_z" for col in valid_score_cols_to_normalize if f"{col}_z" in df_normalized.columns]

        if not analysis_cols_normalized:
            logging.warning(f"No score columns were successfully normalized for {dataset_name}. Skipping analysis.")
            continue

        logging.info(f"Proceeding with analysis for normalized columns: {analysis_cols_normalized}")
        for score_col_norm in analysis_cols_normalized:
            perform_aggregate_subtype_analysis(
                df_aggregated_normalized=df_normalized,
                score_column_normalized=score_col_norm,
                dataset_name=dataset_name,
                min_samples_per_subtype=args.min_samples,
                plot_dir=output_dir if args.save_plots else None,
                save_plots=args.save_plots,
            )

    logging.info("\n--- Aggregate Analysis Complete ---")

if __name__ == "__main__":
    main()