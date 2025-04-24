import argparse
import json
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_uncertainty_df(metrics_dict):
    """
    Parses the results dictionary (from analyze_results.py) into a pandas DataFrame.

    Args:
        metrics_dict (dict): The dictionary loaded from the analysis results JSON/pickle.

    Returns:
        pd.DataFrame: DataFrame with columns ['method', 'metric', 'means']
                      or None if input is invalid.
    """
    if 'uncertainty' not in metrics_dict:
        logging.error("Input dictionary missing 'uncertainty' key.")
        return None

    data = []
    for method_key, method_data in metrics_dict['uncertainty'].items():
        if method_key.endswith('_UNANSWERABLE'):
            continue
        base_method_key = method_key

        for metric_key, metric_values in method_data.items():
            if 'mean' in metric_values:
                mean_val = metric_values['mean']
                data.append([base_method_key, metric_key, mean_val])
            else:
                logging.warning(f"Metric '{metric_key}' for method '{method_key}' missing 'mean' value. Skipping.")

    if not data:
        logging.warning("No valid data extracted from the 'uncertainty' section.")
        return None

    df = pd.DataFrame(data, columns=['method_key', 'metric', 'means'])

    main_methods_map = {
        'semantic_entropy': 'Semantic Entropy',
        'cluster_assignment_entropy': 'Discrete Semantic Entropy',
        'regular_entropy': 'Naive Entropy',
        'p_false_fixed': 'p(True)',             
        'p_ik': 'Embedding Regression'
    }

    df_filtered = df[df['method_key'].isin(main_methods_map.keys())].copy()
    df_filtered['method'] = df_filtered['method_key'].map(main_methods_map)

    original_keys_in_df = df_filtered['method_key'].unique()
    if len(original_keys_in_df) != len(main_methods_map):
         missing_keys = set(main_methods_map.keys()) - set(original_keys_in_df)
         if missing_keys:
              logging.warning(f"Some methods defined in main_methods_map were not found in the results: {missing_keys}")
         extra_keys = set(original_keys_in_df) - set(main_methods_map.keys())
         if extra_keys:
              logging.warning(f"Some methods found in results were not included in main_methods_map: {extra_keys}")

    ordered_methods = list(main_methods_map.values())
    df_filtered['method'] = pd.Categorical(df_filtered['method'], categories=ordered_methods, ordered=True)
    df_filtered = df_filtered.sort_values('method')

    df_final = df_filtered[['method', 'metric', 'means']]

    return df_final

def plot_auroc(df, output_path, run_id=""):
    """
    Generates a bar plot of the AUROC values for each method in the given DataFrame, and saves it to a file.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing the results of the analysis, with columns 'method', 'metric', and 'means'.
        output_path (Path): The path to save the plot to.
        run_id (str, optional): The run ID to include in the plot title. Defaults to an empty string.

    Returns:
        None
    """
    metric_to_plot = 'AUROC'
    plot_df = df[df['metric'] == metric_to_plot].copy()

    if plot_df.empty:
        logging.error(f"No data found for metric '{metric_to_plot}'. Cannot generate plot.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df.plot.bar(x='method', y='means', ax=ax, legend=True)

    ax.set_ylabel(metric_to_plot)
    ax.set_xlabel('Method')
    ax.grid(axis='y', linestyle='-', linewidth=0.5)
    ax.set_ylim(0.1, 1.0)
    ax.set_title(f'AUROC Comparison for Run: {run_id}' if run_id else f'{metric_to_plot} Comparison')
    plt.xticks(rotation=90)
    plt.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Plot saved successfully to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot to {output_path}: {e}")
    finally:
        plt.close(fig)

def main():
    """
    Main entry point for loading analysis results from a JSON file and generating a comparison plot
    for the AUROC metric.

    Args:
        input_json (str): Path to the JSON file containing analysis results (output of analyze_results.py).
        output_plot (str): Path to save the generated AUROC plot.
        run_id (str): Optional Run ID to include in the plot title.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Load analysis results and generate AUROC plot.")
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to the JSON file containing analysis results (output of analyze_results.py)."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="auroc_comparison.png",
        help="Path to save the generated AUROC plot."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Optional Run ID to include in the plot title."
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_plot)

    if not input_path.is_file():
        logging.error(f"Input JSON file not found: {input_path}")
        return

    logging.info(f"Loading analysis results from: {input_path}")
    try:
        with open(input_path, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse JSON file {input_path}: {e}")
        return

    logging.info("Processing results into DataFrame...")
    uncertainty_df = get_uncertainty_df(results_data)

    if uncertainty_df is not None and not uncertainty_df.empty:
        logging.info(f"Generated DataFrame:\n{uncertainty_df.to_string()}")
        logging.info(f"Generating AUROC plot and saving to: {output_path}")
        plot_auroc(uncertainty_df, output_path, run_id=args.run_id)
    else:
        logging.error("Failed to create DataFrame from results. Skipping plot generation.")

if __name__ == "__main__":
    main()