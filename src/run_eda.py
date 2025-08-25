import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from collections import Counter
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_utils import load_ds
from src.utils.utils import load_pickle

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def perform_eda(base_output_dir: Path, data_dir: Path):
    """
    Performs Exploratory Data Analysis on the datasets and model outputs.
    """
    stats = {}

    logging.info("Analyzing dataset characteristics...")
    try:
        stats["bioasq_question_types"] = (
            "The original BioASQ dataset includes types such as 'factoid', 'list', 'yes/no', and 'summary'."
        )

        _, val_bioasq = load_ds("bioasq", seed=42)
        _, val_medquad = load_ds("medquad", seed=42)

        bioasq_lengths = [
            len(ans) for ex in val_bioasq for ans in ex["answers"]["text"]
        ]
        medquad_lengths = [
            len(ans) for ex in val_medquad for ans in ex["answers"]["text"]
        ]

        stats["answer_lengths"] = {
            "bioasq": {
                "mean": np.mean(bioasq_lengths),
                "std": np.std(bioasq_lengths),
                "median": np.median(bioasq_lengths),
            },
            "medquad": {
                "mean": np.mean(medquad_lengths),
                "std": np.std(medquad_lengths),
                "median": np.median(medquad_lengths),
            },
        }
    except Exception as e:
        logging.error(f"Failed to analyze dataset characteristics: {e}", exc_info=True)
        return

    logging.info("Analyzing model error rates...")

    RUN_ID_MAP = {
        "Llama-3.2-1B-BioASQ": "906e2hzj",
        "TinyLlama-v1.1-BioASQ": "rcr1mf25",
        "TinyLlama-med-BioASQ": "7hqvi225",
        "Granite-3.1-1B-BioASQ": "cjghaflw",
        "Llama-3.2-1B-MedQuAD": "drb3w33s",
        "TinyLlama-v1.1-MedQuAD": "m34s7s2x",
        "TinyLlama-med-MedQuAD": "4z43ovtg",
        "Granite-3.1-1B-MedQuAD": "1w0wyfo2",
    }

    stats["error_rates"] = {}
    total_incorrect_outputs = {"bioasq": [], "medquad": []}

    for name, run_id in RUN_ID_MAP.items():
        run_dir = base_output_dir / run_id
        gen_path = run_dir / "validation_generations.pkl"

        if not gen_path.exists():
            logging.warning(
                f"Could not find generations file for {name} ({run_id}) at {gen_path}. Skipping."
            )
            continue

        generations = load_pickle(gen_path)
        if not generations:
            logging.warning(
                f"Failed to load generations for {name} ({run_id}). Skipping."
            )
            continue

        accuracies = [
            details["most_likely_answer"].get("accuracy", 0.0)
            for details in generations.values()
            if "most_likely_answer" in details
        ]

        if not accuracies:
            stats["error_rates"][name] = {
                "incorrect_pct": "N/A",
                "total_incorrect": 0,
                "total_samples": 0,
            }
            continue

        incorrect_count = sum(1 for acc in accuracies if acc <= 0.5)
        total_samples = len(accuracies)
        incorrect_pct = (
            (incorrect_count / total_samples) * 100 if total_samples > 0 else 0
        )

        stats["error_rates"][name] = {
            "incorrect_pct": incorrect_pct,
            "total_incorrect": incorrect_count,
            "total_samples": total_samples,
        }

        if "bioasq" in name.lower():
            total_incorrect_outputs["bioasq"].append(incorrect_count)
        elif "medquad" in name.lower():
            total_incorrect_outputs["medquad"].append(incorrect_count)

    stats["avg_incorrect_outputs"] = {
        "bioasq": (
            np.mean(total_incorrect_outputs["bioasq"])
            if total_incorrect_outputs["bioasq"]
            else 0
        ),
        "medquad": (
            np.mean(total_incorrect_outputs["medquad"])
            if total_incorrect_outputs["medquad"]
            else 0
        ),
    }

    logging.info("Calculating justification stats for N=480...")
    avg_incorrect_bioasq = stats["avg_incorrect_outputs"]["bioasq"]
    avg_incorrect_medquad = stats["avg_incorrect_outputs"]["medquad"]

    stats["n_480_justification"] = {
        "bioasq_pct_of_incorrect": (
            (480 / avg_incorrect_bioasq) * 100 if avg_incorrect_bioasq > 0 else "N/A"
        ),
        "medquad_pct_of_incorrect": (
            (480 / avg_incorrect_medquad) * 100 if avg_incorrect_medquad > 0 else "N/A"
        ),
    }

    print("\n\n" + "=" * 20 + " EDA Results " + "=" * 20)
    print("\n--- Dataset Characteristics ---\n")
    print(f"BioASQ Question Types: {stats['bioasq_question_types']}")
    print(
        f"BioASQ Answer Lengths: Mean={stats['answer_lengths']['bioasq']['mean']:.1f}, Std={stats['answer_lengths']['bioasq']['std']:.1f}, Median={stats['answer_lengths']['bioasq']['median']:.1f}"
    )
    print(
        f"MedQuAD Answer Lengths: Mean={stats['answer_lengths']['medquad']['mean']:.1f}, Std={stats['answer_lengths']['medquad']['std']:.1f}, Median={stats['answer_lengths']['medquad']['median']:.1f}"
    )

    print("\n--- Model Error Rates (SQuAD F1 <= 0.5) ---\n")
    df_errors = pd.DataFrame(stats["error_rates"]).T
    df_errors["incorrect_pct"] = df_errors["incorrect_pct"].apply(
        lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
    )
    print(df_errors)

    print("\n--- Justification for N=480 Sample Size ---\n")
    bioasq_pct_just = stats["n_480_justification"]["bioasq_pct_of_incorrect"]
    medquad_pct_just = stats["n_480_justification"]["medquad_pct_of_incorrect"]

    print(
        f"For BioASQ, N=480 represents on average {bioasq_pct_just:.1f}% of the total incorrect outputs per model."
    )
    print(
        f"For MedQuAD, N=480 represents on average {medquad_pct_just:.1f}% of the total incorrect outputs per model."
    )

    print("\n" + "=" * 53)


if __name__ == "__main__":
    perform_eda(base_output_dir=Path("./outputs/runs"), data_dir=Path("./data"))
