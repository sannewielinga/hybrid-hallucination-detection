import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import logging
from pathlib import Path
from sklearn.metrics import auc
from src.utils.eval_utils import aurac, rejection_accuracy_curve


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
plt.style.use("seaborn-v0_8-darkgrid")


def save_plot(figure, filepath):
    try:
        figure.savefig(filepath, bbox_inches="tight", dpi=150)
        logging.info(f"Plot saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save plot {filepath}: {e}")
    plt.close(figure)


def perform_subtype_analysis(
    df_annotated,
    score_column,
    run_id,
    min_samples_per_subtype=2,
    plot_dir=None,
    save_plots=False,
):

    logging.info(
        f"\n--- Analyzing Score Column: '{score_column}' for Run: {run_id} ---"
    )

    if score_column not in df_annotated.columns:
        logging.error(
            f"Score column '{score_column}' not found in DataFrame. Skipping analysis."
        )
        return
    if df_annotated[score_column].isnull().all():
        logging.warning(
            f"All values in score column '{score_column}' are NaN. Skipping analysis."
        )
        return

    print(f"\nBasic Stats for '{score_column}':")
    print(df_annotated[score_column].describe())

    fig_type, ax_type = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df_annotated,
        x="hallucination_type",
        y=score_column,
        ax=ax_type,
        palette="viridis",
        hue="hallucination_type",
        legend=False
    )
    sns.stripplot(
        data=df_annotated,
        x="hallucination_type",
        y=score_column,
        color=".25",
        size=3,
        alpha=0.5,
        ax=ax_type,
    )
    ax_type.set_title(
        f"{score_column} Distribution by Primary Hallucination Type (Run: {run_id})"
    )
    ax_type.set_xlabel("Primary Hallucination Type")
    ax_type.set_ylabel(f"{score_column} Score")
    ax_type.grid(axis="y", linestyle="--", alpha=0.7)

    if save_plots and plot_dir:
        plot_filename = plot_dir / f"{run_id}_{score_column}_vs_Type.png"
        save_plot(fig_type, plot_filename)
    else:
        try:
            plt.show()
        except UserWarning:
            logging.warning(
                "Plot display not available in current environment. Consider using --save_plots."
            )
        plt.close(fig_type)

    valid_annotations_mask = ~df_annotated['hallucination_type'].isin(['Annotation_Failed', 'Unknown'])
    df_aurac_analysis = df_annotated[valid_annotations_mask].copy()
    df_aurac_analysis = df_aurac_analysis[df_aurac_analysis['hallucination_type'] != 'Other/Unclear'].copy()
    df_aurac_analysis['is_correct_manual'] = ~df_aurac_analysis['hallucination_type'].isin(['Factuality', 'Faithfulness'])

    print(f"\n--- AURAC Calculation for '{score_column}' on Annotated Subset (n={len(df_aurac_analysis)}) ---")
    if not df_aurac_analysis.empty:
        y_true_correct = df_aurac_analysis['is_correct_manual'].values
        y_score_uncertainty = df_aurac_analysis[score_column].values

        valid_metric_mask = ~np.isnan(y_true_correct) & ~np.isnan(y_score_uncertainty)
        if np.sum(valid_metric_mask) > 1:
            aurac_score = aurac(y_true_correct[valid_metric_mask], y_score_uncertainty[valid_metric_mask])
            print(f"AURAC Score: {aurac_score:.4f}")

            if save_plots and plot_dir:
                fractions, accuracies = rejection_accuracy_curve(
                    y_true_correct[valid_metric_mask], y_score_uncertainty[valid_metric_mask]
                )
                fig_rej, ax_rej = plt.subplots(figsize=(8, 6))
                plot_mask = ~np.isnan(accuracies)
                if np.any(plot_mask):
                    ax_rej.plot(fractions[plot_mask], accuracies[plot_mask], marker='o', linestyle='-')
                    ax_rej.set_title(f'Rejection Accuracy Curve for {score_column} (Run: {run_id})')
                    ax_rej.set_xlabel('Fraction of Samples Rejected (Based on Score)')
                    ax_rej.set_ylabel('Accuracy of Accepted Samples')
                    ax_rej.grid(True, linestyle='--')
                    min_acc_plot = np.nanmin(accuracies[plot_mask])
                    ax_rej.set_ylim(bottom=min(0.5, min_acc_plot - 0.05) if not np.isnan(min_acc_plot) else 0.5, top=1.05)
                    ax_rej.text(0.05, 0.1, f'AURAC = {aurac_score:.4f}', transform=ax_rej.transAxes,
                                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                    plot_filename_rej = plot_dir / f"{run_id}_{score_column}_RejectionCurve.png"
                    save_plot(fig_rej, plot_filename_rej)
                else:
                     logging.warning(f"No valid accuracy points to plot rejection curve for {score_column}.")
                     plt.close(fig_rej)

        else:
            print("AURAC Score: Not calculable (insufficient valid data).")
    else:
        print("AURAC Score: Not calculable (no valid annotated samples).")


    logging.info(
        f"\nFiltering subtypes with at least {min_samples_per_subtype} samples for plotting/stats..."
    )
    subtype_counts = df_annotated["hallucination_subtype"].value_counts()
    subtypes_to_plot = subtype_counts[subtype_counts >= min_samples_per_subtype].index
    df_plot_subtype = df_annotated[
        df_annotated["hallucination_subtype"].isin(subtypes_to_plot)
    ].copy()

    if not df_plot_subtype.empty and not df_plot_subtype[score_column].isnull().all():
        fig_subtype, ax_subtype = plt.subplots(figsize=(14, 8))
        median_scores = df_plot_subtype.groupby("hallucination_subtype")[
            score_column
        ].median()
        order = median_scores.sort_values().index

        sns.boxplot(
            data=df_plot_subtype,
            x="hallucination_subtype",
            y=score_column,
            order=order,
            ax=ax_subtype,
            palette="viridis",
            hue="hallucination_subtype",
            legend=False,
        )
        ax_subtype.set_title(
            f"{score_column} Distribution by Hallucination Subtype (Run: {run_id}, n >= {min_samples_per_subtype})"
        )
        ax_subtype.set_xlabel("Hallucination Subtype")
        ax_subtype.set_ylabel(f"{score_column} Score")
        ax_subtype.tick_params(axis="x", rotation=45)
        ax_subtype.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_plots and plot_dir:
            plot_filename = plot_dir / f"{run_id}_{score_column}_vs_Subtype.png"
            save_plot(fig_subtype, plot_filename)
        else:
            try:
                plt.show()
            except UserWarning:
                logging.warning(
                    "Plot display not available in current environment. Consider using --save_plots."
                )
            plt.close(fig_subtype)
    else:
        logging.warning(
            f"Not enough samples per subtype (min={min_samples_per_subtype}) or score column '{score_column}' has only NaNs for subtypes. Skipping subtype plot."
        )

    print(f"\n--- Summary Statistics for {score_column} by Hallucination Type ---")
    print(
        df_annotated.groupby("hallucination_type")[score_column].agg(
            ["mean", "median", "std", "count"]
        )
    )

    if not df_plot_subtype.empty and not df_plot_subtype[score_column].isnull().all():
        print(
            f"\n--- Summary Statistics for {score_column} by Hallucination Subtype (n >= {min_samples_per_subtype}) ---"
        )
        print(
            df_plot_subtype.groupby("hallucination_subtype")[score_column]
            .agg(["mean", "median", "std", "count"])
            .sort_values(by="median")
        )
    else:
        logging.warning(
            f"Skipping subtype summary statistics due to insufficient data or NaN scores in '{score_column}'."
        )

    category1 = "Factuality"
    category2 = "Faithfulness"
    print(
        f"\n--- Mann-Whitney U test comparing {score_column} for '{category1}' vs '{category2}' ---"
    )
    print(f"--- Justification: Comparing grouped primary hallucination categories to maximize power ---")

    df_testable_cat = df_annotated[
        df_annotated['hallucination_type'].isin([category1, category2])
    ].dropna(subset=[score_column])

    if (
        category1 in df_testable_cat["hallucination_type"].unique()
        and category2 in df_testable_cat["hallucination_type"].unique()
    ):
        scores_category1 = df_testable_cat[df_testable_cat["hallucination_type"] == category1][
            score_column
        ]
        scores_category2 = df_testable_cat[df_testable_cat["hallucination_type"] == category2][
            score_column
        ]

        n1, n2 = len(scores_category1), len(scores_category2)
        var1, var2 = np.var(scores_category1), np.var(scores_category2)

        if (n1 > 1 and n2 > 1 and var1 > 0 and var2 > 0):
            try:
                stat, p_value = stats.mannwhitneyu(
                    scores_category1, scores_category2, alternative="two-sided"
                )
                print(
                    f"Comparing {n1} samples ({category1}) vs {n2} samples ({category2})."
                )
                print(f"Statistic: {stat:.3f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("Result: Statistically significant difference (p < 0.05)")
                else:
                    print("Result: No statistically significant difference (p >= 0.05)")
            except ValueError as e:
                print(f"Could not perform Mann-Whitney U test: {e}")
        else:
            print(f"Skipping test: Insufficient samples or zero variance. "
                  f"'{category1}': n={n1}, var={var1:.2E}; "
                  f"'{category2}': n={n2}, var={var2:.2E}.")
    else:
        missing = []
        if category1 not in df_testable_cat["hallucination_type"].unique(): missing.append(category1)
        if category2 not in df_testable_cat["hallucination_type"].unique(): missing.append(category2)
        print(f"Skipping test: One or both categories ({', '.join(missing)}) not found in data.")


    subtype1 = "A3_DiagnosticCriteriaDefinition"
    subtype2 = "B1_ExtrapolationAddition"
    print(
        f"\n--- Mann-Whitney U test comparing {score_column} for '{subtype1}' vs '{subtype2}' ---"
    )
    print(f"--- Justification: Comparing the two most frequent subtypes (Factuality vs. Faithfulness) ---")

    df_testable_sub = df_plot_subtype.dropna(subset=[score_column])

    if (
        subtype1 in df_testable_sub["hallucination_subtype"].unique()
        and subtype2 in df_testable_sub["hallucination_subtype"].unique()
    ):
        scores_subtype1 = df_testable_sub[df_testable_sub["hallucination_subtype"] == subtype1][
            score_column
        ]
        scores_subtype2 = df_testable_sub[df_testable_sub["hallucination_subtype"] == subtype2][
            score_column
        ]

        n1, n2 = len(scores_subtype1), len(scores_subtype2)
        var1, var2 = np.var(scores_subtype1), np.var(scores_subtype2)

        if (n1 > 1 and n2 > 1 and var1 > 0 and var2 > 0):
            try:
                stat, p_value = stats.mannwhitneyu(
                    scores_subtype1, scores_subtype2, alternative="two-sided"
                )
                print(
                    f"Comparing {n1} samples ({subtype1}) vs {n2} samples ({subtype2})."
                )
                print(f"Statistic: {stat:.3f}, P-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("Result: Statistically significant difference (p < 0.05)")
                else:
                    print("Result: No statistically significant difference (p >= 0.05)")
            except ValueError as e:
                print(f"Could not perform Mann-Whitney U test: {e}")
        else:
            print(f"Skipping test: Insufficient samples or zero variance. "
                  f"'{subtype1}': n={n1}, var={var1:.2E}; "
                  f"'{subtype2}': n={n2}, var={var2:.2E}.")
    else:
        missing = []
        if subtype1 not in df_testable_sub["hallucination_subtype"].unique(): missing.append(subtype1)
        if subtype2 not in df_testable_sub["hallucination_subtype"].unique(): missing.append(subtype2)
        print(f"Skipping test: One or both subtypes ({', '.join(missing)}) not found in filtered data (n >= {min_samples_per_subtype}).")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hallucination subtypes vs uncertainty scores."
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the final merged analysis data CSV file (containing annotations and scores).",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Identifier for the experimental run (used for titles/filenames).",
    )
    parser.add_argument(
        "--score_columns",
        type=str,
        nargs="+",
        default=["semantic_entropy"],
        help="List of score column names to analyze (e.g., semantic_entropy internal_signal_score hybrid_simple_score).",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=2,
        help="Minimum samples required per subtype for detailed plotting and stats.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="./analysis_plots",
        help="Directory to save output plots.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots to files instead of displaying them.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.is_file():
        logging.error(f"Input CSV file not found: {input_path}")
        return

    logging.info(f"Loading final analysis data from: {input_path}")
    try:
        df_full = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df_full)} total rows.")
        print("Columns:", df_full.columns.tolist())
        print("\nBasic Info:")
        df_full.info()
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        return

    required_annotation_cols = ["hallucination_type", "hallucination_subtype"]
    df_annotated = df_full.dropna(subset=required_annotation_cols).copy()
    num_annotated = len(df_annotated)
    logging.info(
        f"\nUsing {num_annotated} rows with complete type/subtype annotations for analysis."
    )

    if num_annotated == 0:
        logging.error(
            "No rows with complete annotations found. Cannot perform analysis."
        )
        return

    print("\nHallucination Type Counts:")
    print(df_annotated["hallucination_type"].value_counts())
    print("\nHallucination Subtype Counts:")
    print(df_annotated["hallucination_subtype"].value_counts())

    plot_dir = Path(args.plot_dir)
    if args.save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Plots will be saved to: {plot_dir}")

    for score_col in args.score_columns:
        perform_subtype_analysis(
            df_annotated=df_annotated,
            score_column=score_col,
            run_id=args.run_id,
            min_samples_per_subtype=args.min_samples,
            plot_dir=plot_dir if args.save_plots else None,
            save_plots=args.save_plots,
        )

    logging.info("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()