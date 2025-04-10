import pandas as pd
import argparse
import pickle
import logging
import numpy as np
from pathlib import Path

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


def process_raw_annotations(raw_csv_path):
    """Loads raw annotations, merges subtype columns. Returns DataFrame or None."""
    raw_path = Path(raw_csv_path)

    if not raw_path.is_file():
        logging.error(f"Raw annotation file not found at {raw_path}")
        return None

    logging.info(f"Processing raw annotations from: {raw_path}")
    try:
        df = pd.read_csv(raw_path)
        logging.info(f"Loaded {len(df)} rows from raw annotations export.")

        if "hallucination_type" not in df.columns:
            logging.warning(
                "'hallucination_type' column missing. Will attempt to infer."
            )
            df["hallucination_type"] = pd.NA
        if "hallucination_subtype" not in df.columns:
            df["hallucination_subtype"] = pd.NA

        processed_factuality = 0
        processed_faithfulness = 0

        if "factuality_subtype_choice" in df.columns:
            mask_infer_factuality = (
                df["factuality_subtype_choice"].notna()
                & df["hallucination_type"].isna()
            )
            df.loc[mask_infer_factuality, "hallucination_type"] = "Factuality"
            logging.info(
                f"Inferred 'Factuality' for {mask_infer_factuality.sum()} rows based on 'factuality_subtype_choice'."
            )

            mask_use_factuality = df["factuality_subtype_choice"].notna() & (
                df["hallucination_type"] == "Factuality"
            )
            df.loc[mask_use_factuality, "hallucination_subtype"] = df.loc[
                mask_use_factuality, "factuality_subtype_choice"
            ]
            processed_factuality = mask_use_factuality.sum()
        else:
            logging.warning("'factuality_subtype_choice' column not found.")

        if "faithfulness_subtype_choice" in df.columns:
            mask_infer_faithfulness = (
                df["faithfulness_subtype_choice"].notna()
                & df["hallucination_type"].isna()
            )
            df.loc[mask_infer_faithfulness, "hallucination_type"] = "Faithfulness"
            logging.info(
                f"Inferred 'Faithfulness' for {mask_infer_faithfulness.sum()} rows based on 'faithfulness_subtype_choice'."
            )

            mask_use_faithfulness = df["faithfulness_subtype_choice"].notna() & (
                df["hallucination_type"] == "Faithfulness"
            )
            df.loc[mask_use_faithfulness, "hallucination_subtype"] = df.loc[
                mask_use_faithfulness, "faithfulness_subtype_choice"
            ]
            processed_faithfulness = mask_use_faithfulness.sum()
        else:
            logging.warning("'faithfulness_subtype_choice' column not found.")

        if "annotation_id" in df.columns:
            mask_infer_other = (
                df["hallucination_type"].isna() & df["annotation_id"].notna()
            )
            df.loc[mask_infer_other, "hallucination_type"] = "Other_Unclear"
            logging.info(
                f"Assigned 'Other_Unclear' to {mask_infer_other.sum()} annotated rows where type couldn't be inferred."
            )

        logging.info(f"Processed {processed_factuality} factuality subtypes.")
        logging.info(f"Processed {processed_faithfulness} faithfulness subtypes.")

        columns_needed = [
            "id",
            "question",
            "context",
            "reference_answers",
            "generated_answer",
            "is_correct",
            "accuracy_metric_score",
            "hallucination_type",
            "hallucination_subtype",
            "annotation_notes",
            "high_temp_samples",
            "selected_for_annotation",
        ]
        columns_present = [col for col in columns_needed if col in df.columns]
        missing_cols = set(columns_needed) - set(columns_present)
        if missing_cols:
            logging.warning(
                f"Columns missing from raw CSV, won't be in output: {missing_cols}"
            )

        processed_df = df[columns_present].copy()
        logging.info("Finished processing annotations.")
        return processed_df

    except FileNotFoundError:
        logging.error(f"Raw annotation file not found at {raw_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing annotations: {e}")
        return None


def load_all_uncertainty_scores(run_dir_path):
    """Loads SE and other scores for ALL validation samples from pkl files."""
    run_dir = Path(run_dir_path)
    uncertainty_path = run_dir / "uncertainty_measures.pkl"
    generations_path = run_dir / "validation_generations.pkl"

    uncertainty_data = load_pickle(uncertainty_path)
    generations_data = load_pickle(generations_path)

    if uncertainty_data is None or generations_data is None:
        logging.error(
            "Cannot load uncertainty scores without uncertainty AND generation pkl data."
        )
        return None

    logging.info(
        "Loading ALL uncertainty scores (SE, Naive, etc.) and matching to IDs..."
    )
    try:
        uncertainty_measures = uncertainty_data.get("uncertainty_measures", {})
        se_scores = uncertainty_measures.get("semantic_entropy")
        naive_entropy = uncertainty_measures.get("regular_entropy")

        if se_scores is None:
            logging.error("Missing 'semantic_entropy' in uncertainty_measures.pkl")
            return None

        task_ids_ordered = list(generations_data.keys())
        num_tasks = len(task_ids_ordered)
        num_se = len(se_scores)

        if num_tasks != num_se:
            logging.warning(
                f"Mismatch length: {num_tasks} tasks vs {num_se} SE scores. Using min length: {min(num_tasks, num_se)}"
            )
            min_len = min(num_tasks, num_se)
        else:
            min_len = num_tasks

        data_dict = {
            "id": task_ids_ordered[:min_len],
            "semantic_entropy": se_scores[:min_len],
        }

        if naive_entropy is not None and len(naive_entropy) >= min_len:
            data_dict["naive_entropy"] = naive_entropy[:min_len]

        all_scores_df = pd.DataFrame(data_dict)
        logging.info(f"Loaded ALL scores for {len(all_scores_df)} tasks.")
        return all_scores_df

    except Exception as e:
        logging.error(f"Error loading all uncertainty scores: {e}")
        return None


def load_internal_signal_results(is_results_csv_path):
    """Loads internal signal probe results (test split only)."""
    is_path = Path(is_results_csv_path)
    if not is_path.is_file():
        logging.error(f"Internal signal results file not found: {is_path}")
        return None

    logging.info(f"Loading internal signal results from: {is_path}")
    try:
        is_df = pd.read_csv(is_path)
        cols_to_keep = [
            "id",
            "internal_signal_score",
            "se_normalized",
            "is_normalized",
            "hybrid_0.5_0.5",
        ]
        cols_present = [col for col in cols_to_keep if col in is_df.columns]
        if not cols_present or "id" not in cols_present:
            logging.error(
                f"Essential columns ('id', etc.) not found in IS results file: {is_path}"
            )
            return None
        logging.info(
            f"Loaded {len(is_df)} rows and columns {cols_present} from internal signal results."
        )
        return is_df[cols_present]
    except Exception as e:
        logging.error(f"Error loading internal signal results CSV: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process annotations, load scores, and merge into final analysis file."
    )
    parser.add_argument(
        "raw_annotations_csv",
        help="Path to the raw CSV file exported from Label Studio.",
    )
    parser.add_argument(
        "run_files_dir",
        help="Path to the directory containing pkl files (validation_generations.pkl, uncertainty_measures.pkl) for the specific run.",
    )
    parser.add_argument(
        "internal_signal_csv",
        help="Path to the CSV file containing internal signal results (output of internal_signal_probe.py).",
    )
    parser.add_argument(
        "final_output_csv", help="Path to save the final merged CSV for analysis."
    )
    args = parser.parse_args()

    processed_annotations_df = process_raw_annotations(args.raw_annotations_csv)
    if processed_annotations_df is None:
        exit()

    all_uncertainty_scores_df = load_all_uncertainty_scores(args.run_files_dir)
    if all_uncertainty_scores_df is None:
        exit()

    internal_signal_df = load_internal_signal_results(args.internal_signal_csv)
    if internal_signal_df is None:
        exit()  # Or proceed without IS scores if desired

    logging.info("Merging processed annotations with ALL uncertainty scores...")

    processed_annotations_df["id"] = (
        processed_annotations_df["id"].astype(str).str.strip()
    )
    all_uncertainty_scores_df["id"] = (
        all_uncertainty_scores_df["id"].astype(str).str.strip()
    )
    internal_signal_df["id"] = internal_signal_df["id"].astype(str).str.strip()

    logging.info(f"Annotation DF head:\n{processed_annotations_df.head()}")
    logging.info(
        f"Annotation DF IDs unique: {processed_annotations_df['id'].nunique()}, total: {len(processed_annotations_df)}"
    )
    logging.info(f"All Scores DF head:\n{all_uncertainty_scores_df.head()}")
    logging.info(
        f"All Scores DF IDs unique: {all_uncertainty_scores_df['id'].nunique()}, total: {len(all_uncertainty_scores_df)}"
    )

    merged_df = pd.merge(
        processed_annotations_df, all_uncertainty_scores_df, on="id", how="left"
    )
    logging.info(f"After merge with All Scores - Head:\n{merged_df.head()}")
    logging.info(
        f"After merge with All Scores - NaN SE count: {merged_df['semantic_entropy'].isnull().sum()}"
    )

    if internal_signal_df is not None:
        logging.info(f"IS Scores DF head:\n{internal_signal_df.head()}")
        logging.info(
            f"IS Scores DF IDs unique: {internal_signal_df['id'].nunique()}, total: {len(internal_signal_df)}"
        )
        final_df = pd.merge(merged_df, internal_signal_df, on="id", how="left")
        logging.info(f"After merge with IS Scores - Head:\n{final_df.head()}")
        logging.info(
            f"After merge with IS Scores - NaN SE count: {final_df['semantic_entropy'].isnull().sum()}"
        )
        logging.info(
            f"After merge with IS Scores - NaN IS count: {final_df['internal_signal_score'].isnull().sum()}"
        )
    else:
        final_df = merged_df

    final_df = pd.merge(merged_df, internal_signal_df, on="id", how="left")
    logging.info(f"Rows after merging with internal signal scores: {len(final_df)}")
    if "internal_signal_score" in final_df:
        logging.info(
            f"Number of rows with non-null internal signal scores: {final_df['internal_signal_score'].notna().sum()}"
        )
        if final_df["internal_signal_score"].notna().sum() == 0:
            logging.warning(
                "No internal signal scores were merged. Check IDs in internal_signal_results.csv match annotations."
            )

    final_path = Path(args.final_output_csv)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    annotated_count = len(final_df["hallucination_type"].dropna())
    logging.info(
        f"Final dataset includes {annotated_count} rows with hallucination type assigned."
    )

    final_df.to_csv(final_path, index=False)
    logging.info(f"Final analysis-ready data saved to: {final_path}")
    logging.info("\n--- Processing Complete ---")
