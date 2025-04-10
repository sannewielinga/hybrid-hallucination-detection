from collections import defaultdict
import logging
import os
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch

from src.utils.data_utils import load_ds
from src.utils.p_ik import get_p_ik
from src.utils.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy,
    predictive_entropy_rao,
)
from src.utils.semantic_entropy import (
    cluster_assignment_entropy,
    context_entails_response,
    EntailmentDeberta,
)
from src.utils.semantic_entropy import (
    EntailmentGPT4,
    EntailmentGPT35,
    EntailmentGPT4Turbo,
    EntailmentLlama,
)
from src.utils import p_true as p_true_utils
from src.utils import utils
from src.utils import logging_utils

EXP_DETAILS_FILE = "experiment_details.pkl"
VALIDATION_GEN_FILE = "validation_generations.pkl"
TRAIN_GEN_FILE = "train_generations.pkl"
OUTPUT_UNCERTAINTY_FILE = "uncertainty_measures.pkl"


def main(args):
    logging_utils.setup_logger()
    run_dir = Path(args.run_dir)
    run_id = run_dir.name
    logging.info(f"--- Starting SE Computation for Run: {run_id} ---")
    logging.info(f"Reading inputs from: {run_dir}")
    logging.info(f"Full args received: {args}")

    validation_generations_path = run_dir / VALIDATION_GEN_FILE
    validation_generations = utils.load_pickle(validation_generations_path)
    if validation_generations is None:
        logging.error(
            f"Failed to load {validation_generations_path}. Cannot compute SE."
        )
        return

    train_generations = None
    if args.compute_p_ik or args.compute_p_ik_answerable:
        train_generations_path = run_dir / TRAIN_GEN_FILE
        train_generations = utils.load_pickle(train_generations_path)
        if train_generations is None:
            logging.warning(
                f"Train generations file {train_generations_path} not found, cannot compute P(IK)."
            )
            args.compute_p_ik = False
            args.compute_p_ik_answerable = False

    entailment_model = None
    if args.compute_predictive_entropy:
        logging.info(
            "Beginning loading for entailment model: %s", args.entailment_model
        )
        try:
            if (
                "BiomedNLP-PubMedBERT" in args.entailment_model
                or args.entailment_model == "deberta"
            ):
                logging.info(
                    f"Using EntailmentDeberta class for {args.entailment_model}"
                )
                entailment_model = EntailmentDeberta(model_name=args.entailment_model)
            elif "llama" in args.entailment_model.lower():
                entailment_model = EntailmentLlama(
                    args.entailment_cache_id,
                    args.entailment_cache_only,
                    args.entailment_model,
                )
            else:
                logging.error(
                    f"Unsupported entailment model type specified: {args.entailment_model}"
                )
                return
            if entailment_model:
                logging.info("Entailment model loading complete.")
        except Exception as e:
            logging.error(
                f"Failed to load entailment model '{args.entailment_model}': {e}",
                exc_info=True,
            )
            return
    else:
        logging.info("Skipping predictive entropy calculation based on args.")
        entailment_model = None

    metric = None
    if args.recompute_accuracy:
        logging.warning("Recompute accuracy enabled.")
        metric = utils.get_metric(args.metric)

    result_dict = {"uncertainty_measures": defaultdict(list), "semantic_ids": []}
    entropies = result_dict["uncertainty_measures"]
    validation_embeddings = []
    validation_is_true = []
    validation_answerable = []
    count = 0
    total_samples_to_process = len(validation_generations)
    samples_to_process = list(validation_generations.keys())

    logging.info(
        f"Starting calculation loop for {min(args.num_eval_samples, total_samples_to_process)} validation samples..."
    )

    def is_answerable(generation):
        ref = generation.get("reference", {})
        ans = ref.get("answers", {})
        text = ans.get("text", [])
        return bool(text)

    for idx, tid in enumerate(samples_to_process):
        if idx >= args.num_eval_samples:
            logging.info(
                f"Reached num_eval_samples limit ({args.num_eval_samples}). Stopping."
            )
            break

        example = validation_generations[tid]
        most_likely_answer = example.get("most_likely_answer", {})
        accuracy = most_likely_answer.get("accuracy")

        if args.recompute_accuracy and metric is not None:
            response_text = most_likely_answer.get("response")
            if response_text and response_text != "[PREDICTION FAILED]":
                try:
                    acc = metric(response_text, example, None)
                except Exception as e:
                    logging.warning(f"Metric recalculation failed for ID {tid}: {e}")
                    acc = 0.0
            else:
                acc = 0.0
            accuracy = acc
        elif accuracy is None:
            logging.warning(f"Accuracy missing for ID {tid}. Assuming incorrect.")
            accuracy = 0.0

        is_correct = accuracy > args.probe_accuracy_threshold
        validation_is_true.append(is_correct)

        if args.compute_p_ik or args.compute_p_ik_answerable:
            embedding = most_likely_answer.get("embedding")
            if embedding is not None:
                if isinstance(embedding, torch.Tensor):
                    validation_embeddings.append(embedding.cpu())
                else:
                    try:
                        validation_embeddings.append(torch.tensor(embedding).cpu())
                    except:
                        logging.warning(
                            "Could not convert embedding to tensor for ID %s", tid
                        )

        validation_answerable.append(is_answerable(example))

        if args.compute_predictive_entropy and entailment_model:
            full_responses = example.get("responses", [])
            log_liks = [
                r[1]
                for r in full_responses
                if isinstance(r, (list, tuple)) and len(r) > 1 and r[1] is not None
            ]
            responses = [
                r[0]
                for r in full_responses
                if isinstance(r, (list, tuple)) and len(r) > 0
            ]

            semantic_entropy_val = np.nan
            regular_entropy_val = np.nan
            cluster_entropy_val = np.nan
            semantic_ids_val = []

            if log_liks and responses and len(log_liks) == len(responses):
                try:
                    semantic_ids = get_semantic_ids(
                        responses,
                        model=entailment_model,
                        strict_entailment=args.strict_entailment,
                        example=example,
                    )
                    semantic_ids_val = semantic_ids
                    log_liks_agg = [np.mean(ll) if ll else np.nan for ll in log_liks]

                    if not np.isnan(log_liks_agg).any():
                        cluster_entropy_val = cluster_assignment_entropy(semantic_ids)
                        regular_entropy_val = predictive_entropy(log_liks_agg)
                        log_likelihood_per_semantic_id = logsumexp_by_id(
                            semantic_ids, log_liks_agg, agg="sum_normalized"
                        )
                        semantic_entropy_val = predictive_entropy_rao(
                            log_likelihood_per_semantic_id
                        )
                    else:
                        logging.warning(
                            f"NaN found in aggregated log_liks for ID {tid}. Storing NaN for entropies."
                        )

                except Exception as e:
                    logging.error(f"Error calculating SE for {tid}: {e}", exc_info=True)

            else:
                logging.warning(
                    f"Skipping SE calculation for ID {tid} due to missing/mismatched responses/log_liks."
                )

            entropies["semantic_entropy"].append(semantic_entropy_val)
            entropies["regular_entropy"].append(regular_entropy_val)
            entropies["cluster_assignment_entropy"].append(cluster_entropy_val)
            result_dict["semantic_ids"].append(semantic_ids_val)

        count += 1

    logging.info(f"Finished processing loop. Samples processed: {count}")

    result_dict["validation_is_false"] = [not b for b in validation_is_true]
    result_dict["validation_unanswerable"] = [not b for b in validation_answerable]

    if (args.compute_p_ik or args.compute_p_ik_answerable) and train_generations:
        if len(validation_embeddings) != count:
            logging.warning(
                f"P(IK) validation embedding count ({len(validation_embeddings)}) doesn't match processed samples ({count}). P(IK) results might be misaligned or incomplete."
            )

        if args.compute_p_ik or args.compute_p_ik_answerable:
            train_is_true, train_embeddings, train_answerable = [], [], []
            logging.info("Preparing P(IK) training data...")
            for tid_train, train_example in train_generations.items():
                train_most_likely = train_example.get("most_likely_answer", {})
                train_acc = train_most_likely.get("accuracy")
                train_emb = train_most_likely.get("embedding")
                if train_acc is not None and train_emb is not None:
                    train_is_true.append(train_acc > args.probe_accuracy_threshold)
                    train_answerable.append(is_answerable(train_example))
                    if isinstance(train_emb, torch.Tensor):
                        train_embeddings.append(train_emb.cpu())
                    else:
                        try:
                            train_embeddings.append(torch.tensor(train_emb).cpu())
                        except:
                            logging.warning(
                                f"Skipping training embedding for {tid_train}, cannot convert."
                            )
                else:
                    logging.warning(
                        f"Skipping P(IK) training sample {tid_train}, missing accuracy or embedding."
                    )

            train_is_false = [not b for b in train_is_true]
            train_unanswerable = [not b for b in train_answerable]

            if train_embeddings and validation_embeddings:
                if args.compute_p_ik:
                    logging.info("Training P(IK) for correctness...")
                    p_ik_preds = get_p_ik(
                        train_embeddings=train_embeddings,
                        is_false=train_is_false,
                        eval_embeddings=validation_embeddings,
                        eval_is_false=result_dict["validation_is_false"],
                    )
                    result_dict["uncertainty_measures"]["p_ik"] = p_ik_preds.tolist()
                if args.compute_p_ik_answerable:
                    logging.info("Training P(IK) for answerability...")
                    p_ik_ans_preds = get_p_ik(
                        train_embeddings=train_embeddings,
                        is_false=train_unanswerable,
                        eval_embeddings=validation_embeddings,
                        eval_is_false=result_dict["validation_unanswerable"],
                    )
                    result_dict["uncertainty_measures"][
                        "p_ik_unanswerable"
                    ] = p_ik_ans_preds.tolist()
            else:
                logging.warning(
                    "Skipping P(IK) calculation due to missing train or validation embeddings."
                )

    output_path = run_dir / OUTPUT_UNCERTAINTY_FILE
    try:
        final_uncertainty_measures = {}
        for key, score_list in result_dict["uncertainty_measures"].items():
            if len(score_list) == count:
                final_uncertainty_measures[key] = score_list
            else:
                logging.warning(
                    f"Length mismatch for score '{key}'. Expected {count}, got {len(score_list)}. This score will NOT be saved."
                )
        result_dict["uncertainty_measures"] = final_uncertainty_measures

        if "semantic_ids" in result_dict and len(result_dict["semantic_ids"]) != count:
            logging.warning(
                f"Length mismatch for 'semantic_ids'. Expected {count}, got {len(result_dict['semantic_ids'])}. Semantic IDs might be incomplete."
            )
        with open(output_path, "wb") as f:
            pickle.dump(result_dict, f)
        logging.info(f"Final uncertainty measures saved to: {output_path}")
    except Exception as e:
        logging.error(
            f"Failed to save final uncertainty measures to {output_path}: {e}"
        )

    logging.info(f"--- SE Computation Stage Complete for Run: {run_id} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Semantic Entropy and other uncertainty measures."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory containing run files (pkls).",
    )
    parser.add_argument("--entailment_model", type=str, default="deberta")
    parser.add_argument("--strict_entailment", action="store_true")
    parser.add_argument(
        "--no-strict_entailment", dest="strict_entailment", action="store_false"
    )
    parser.set_defaults(strict_entailment=True)

    parser.add_argument("--condition_on_question", action="store_true")
    parser.add_argument(
        "--no-condition_on_question", dest="condition_on_question", action="store_false"
    )
    parser.set_defaults(condition_on_question=True)

    parser.add_argument("--use_num_generations", type=int, default=-1)
    parser.add_argument(
        "--use_all_generations", dest="use_all_generations_se", action="store_true"
    )
    parser.add_argument(
        "--no-use_all_generations", dest="use_all_generations_se", action="store_false"
    )
    parser.set_defaults(use_all_generations_se=True)

    parser.add_argument("--num_eval_samples", type=int, default=int(1e19))
    parser.add_argument("--compute_predictive_entropy", action="store_true")
    parser.add_argument(
        "--no-compute_predictive_entropy",
        dest="compute_predictive_entropy",
        action="store_false",
    )
    parser.set_defaults(compute_predictive_entropy=True)

    parser.add_argument("--compute_p_ik", action="store_true", default=False)
    parser.add_argument("--compute_p_ik_answerable", action="store_true", default=False)
    parser.add_argument("--recompute_accuracy", action="store_true", default=False)
    parser.add_argument(
        "--metric", default="squad", help="Metric for recomputing accuracy"
    )
    parser.add_argument(
        "--probe_accuracy_threshold",
        type=float,
        default=0.5,
        help="Threshold for 'is_correct' label",
    )

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--entailment_cache_id", default=None)
    parser.add_argument("--entailment_cache_only", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.error(f"Unknown args passed to compute_se.py: {unknown}")
    logging.info(f"Args received by compute_se.py: {args}")

    main(args)
