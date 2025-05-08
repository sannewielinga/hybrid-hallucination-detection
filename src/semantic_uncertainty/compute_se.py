from collections import defaultdict
import logging
import os
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from src.utils.data_utils import load_ds
from src.utils.p_ik import get_p_ik
from src.utils.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao,
    cluster_assignment_entropy,
    context_entails_response,
    EntailmentDeberta,
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
    logging.info(f"--- Starting SE & NSE Computation for Run: {run_id} ---")
    logging.info(f"Reading inputs from: {run_dir}")
    logging.info(f"Full args received: {args}")

    validation_generations_path = run_dir / VALIDATION_GEN_FILE
    validation_generations = utils.load_pickle(validation_generations_path)
    if validation_generations is None:
        logging.error(f"Failed to load {validation_generations_path}. Cannot compute SE/NSE.")
        return

    train_generations = None
    if args.compute_p_ik or args.compute_p_ik_answerable:
        train_generations_path = run_dir / TRAIN_GEN_FILE
        train_generations = utils.load_pickle(train_generations_path)
        if train_generations is None:
            logging.warning(f"Train generations file {train_generations_path} not found, cannot compute P(IK).")
            args.compute_p_ik = False
            args.compute_p_ik_answerable = False

    entailment_model = None
    if args.compute_predictive_entropy:
        logging.info("Beginning loading for entailment model: %s", args.entailment_model)
        try:
            if ("BiomedNLP-PubMedBERT" in args.entailment_model or args.entailment_model == "deberta"):
                logging.info(f"Using EntailmentDeberta class for {args.entailment_model}")
                entailment_model_name = args.entailment_model if args.entailment_model != "deberta" else "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"
                entailment_model = EntailmentDeberta(model_name=entailment_model_name)
            elif "llama" in args.entailment_model.lower():
                 entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
            else:
                logging.error(f"Unsupported entailment model type specified: {args.entailment_model}")
                return
            if entailment_model:
                logging.info("Entailment model loading complete.")
        except Exception as e:
            logging.error(f"Failed to load entailment model '{args.entailment_model}': {e}", exc_info=True)
            return
    else:
        logging.info("Skipping SE, NSE, and other entropy calculations based on args.")
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

    logging.info(f"Starting calculation loop for {min(args.num_eval_samples, total_samples_to_process)} validation samples...")

    def is_answerable(generation):
        ref = generation.get("reference", {})
        ans = ref.get("answers", {})
        text = ans.get("text", [])
        return bool(text)

    for idx, tid in enumerate(samples_to_process):
        if idx >= args.num_eval_samples:
            logging.info(f"Reached num_eval_samples limit ({args.num_eval_samples}). Stopping.")
            break

        example = validation_generations[tid]
        most_likely_answer = example.get("most_likely_answer", {})

        metric_score_raw = most_likely_answer.get("accuracy_metric_score")
        is_correct_derived = most_likely_answer.get("is_correct")

        if args.recompute_accuracy and metric is not None:
            response_text = most_likely_answer.get("response")
            if response_text and response_text != "[PREDICTION FAILED]":
                try:
                    metric_score_raw = metric(response_text, example, None)
                    is_correct_derived = metric_score_raw > args.metric_threshold
                except Exception as e:
                    logging.warning(f"Metric recalculation failed for ID {tid}: {e}")
                    metric_score_raw = 0.0; is_correct_derived = False
            else:
                 metric_score_raw = 0.0; is_correct_derived = False
        elif is_correct_derived is None:
            if metric_score_raw is not None and not pd.isna(metric_score_raw):
                 logging.warning(f"Correctness label missing for ID {tid}, deriving from score using threshold {args.metric_threshold}.")
                 is_correct_derived = metric_score_raw > args.metric_threshold
            else:
                 logging.warning(f"Accuracy score missing for ID {tid}. Assuming incorrect."); is_correct_derived = False

        validation_is_true.append(is_correct_derived)

        if args.compute_p_ik or args.compute_p_ik_answerable:
            embedding_data_obj = most_likely_answer.get("embedding")

            embedding_for_pik = None
            if isinstance(embedding_data_obj, dict):
                if 'layer_-1' in embedding_data_obj and embedding_data_obj['layer_-1'] is not None:
                    embedding_for_pik = embedding_data_obj['layer_-1']
                else:
                    logging.warning(f"P(IK) for {tid}: 'layer_-1' not found or None in embedding dict. Looking for other layers.")
                    potential_keys = [k for k,v in embedding_data_obj.items() if k.startswith("layer_") and v is not None]
                    if potential_keys:
                        try:
                            sorted_fallback_keys = sorted(potential_keys, key=lambda x: int(x.split('_')[1]))
                            if sorted_fallback_keys:
                                embedding_for_pik = embedding_data_obj[sorted_fallback_keys[-1]]
                                logging.info(f"P(IK) for {tid}: Using {sorted_fallback_keys[-1]} as fallback.")
                        except (ValueError, IndexError):
                            if potential_keys:
                                 embedding_for_pik = embedding_data_obj[sorted(potential_keys)[0]]
                                 logging.info(f"P(IK) for {tid}: Using alphanumerically first key {sorted(potential_keys)[0]} as fallback.")
            elif embedding_data_obj is not None:
                embedding_for_pik = embedding_data_obj

            if embedding_for_pik is not None:
                try:
                    if isinstance(embedding_for_pik, torch.Tensor):
                        validation_embeddings.append(embedding_for_pik.cpu().flatten())
                    else:
                        validation_embeddings.append(torch.tensor(embedding_for_pik).cpu().flatten())
                except Exception as e_conv_pik:
                    logging.warning(f"Could not convert/append embedding to tensor for P(IK) for ID {tid}: {e_conv_pik}")
            else:
                logging.warning(f"P(IK) for {tid}: No suitable embedding found after checking dict/old format.")

        validation_answerable.append(is_answerable(example))

        if args.compute_predictive_entropy and entailment_model:
            full_responses = example.get("responses", [])
            log_liks = [r[1] for r in full_responses if isinstance(r, (list, tuple)) and len(r) > 1 and r[1] is not None]
            responses = [r[0] for r in full_responses if isinstance(r, (list, tuple)) and len(r) > 0]

            se_val = np.nan
            normalized_se_val = np.nan
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

                    log_liks_agg = []
                    for ll_list in log_liks:
                        if ll_list and all(isinstance(x, (int, float)) for x in ll_list):
                            log_liks_agg.append(np.mean(ll_list))
                        else:
                            log_liks_agg.append(np.nan)

                    if not np.isnan(log_liks_agg).any():
                        cluster_entropy_val = cluster_assignment_entropy(semantic_ids)
                        regular_entropy_val = predictive_entropy_rao(log_liks_agg)

                        log_likelihood_per_semantic_id = logsumexp_by_id(
                            semantic_ids, log_liks_agg, agg="sum_normalized"
                        )
                        num_clusters = len(log_likelihood_per_semantic_id)

                        if num_clusters > 0:
                            se_val = predictive_entropy_rao(log_likelihood_per_semantic_id)

                            if num_clusters <= 1:
                                normalized_se_val = 0.0
                            else:
                                max_entropy = np.log(num_clusters)
                                if max_entropy > 1e-9 and not np.isnan(se_val):
                                     normalized_se_val = se_val / max_entropy
                                else:
                                     normalized_se_val = np.nan

                        else:
                            se_val = np.nan
                            normalized_se_val = np.nan
                    else:
                        logging.warning(f"NaN found in aggregated log_liks for ID {tid}. Storing NaN for entropies.")

                except Exception as e:
                    logging.error(f"Error calculating entropies for {tid}: {e}", exc_info=True)

            else:
                logging.warning(f"Skipping entropy calculations for ID {tid} due to missing/mismatched responses/log_liks.")

            entropies["semantic_entropy"].append(se_val)
            entropies["normalized_semantic_entropy"].append(normalized_se_val)
            entropies["regular_entropy"].append(regular_entropy_val)
            entropies["cluster_assignment_entropy"].append(cluster_entropy_val)
            result_dict["semantic_ids"].append(semantic_ids_val)

        count += 1

    logging.info(f"Finished processing loop. Samples processed: {count}")

    result_dict["validation_is_false"] = [not b for b in validation_is_true]
    result_dict["validation_unanswerable"] = [not b for b in validation_answerable]

    if (args.compute_p_ik or args.compute_p_ik_answerable) and train_generations:
        if len(validation_embeddings) != count:
             logging.warning(f"P(IK) validation embedding count ({len(validation_embeddings)}) doesn't match processed samples ({count}). P(IK) results might be misaligned or incomplete.")

        train_is_true, train_embeddings, train_answerable = [], [], []
        logging.info("Preparing P(IK) training data...")
        if isinstance(train_generations, dict):
            train_items = train_generations.items()
        else:
            logging.error("train_generations is not a dictionary. Cannot prepare P(IK) training data.")
            train_items = []

        for tid_train, train_example in train_items:
            train_most_likely = train_example.get("most_likely_answer", {})
            train_metric_score_raw = train_most_likely.get("accuracy_metric_score")
            train_is_correct_derived = train_most_likely.get("is_correct")
            train_emb_data_obj = train_most_likely.get("embedding")

            if train_is_correct_derived is None:
                 if train_metric_score_raw is not None and not pd.isna(train_metric_score_raw):
                     train_is_correct_derived = train_metric_score_raw > args.metric_threshold
                 else:
                     train_is_correct_derived = False

            embedding_for_pik_train = None
            if isinstance(train_emb_data_obj, dict):
                if 'layer_-1' in train_emb_data_obj and train_emb_data_obj['layer_-1'] is not None:
                    embedding_for_pik_train = train_emb_data_obj['layer_-1']
                else:
                    logging.warning(f"P(IK) Train for {tid_train}: 'layer_-1' missing/None. Fallback needed.")
                    potential_keys_train = [k for k,v in train_emb_data_obj.items() if k.startswith("layer_") and v is not None]
                    if potential_keys_train:
                        try:
                            sorted_fallback_keys_train = sorted(potential_keys_train, key=lambda x: int(x.split('_')[1]))
                            if sorted_fallback_keys_train: embedding_for_pik_train = train_emb_data_obj[sorted_fallback_keys_train[-1]]
                        except (ValueError, IndexError):
                            if potential_keys_train: embedding_for_pik_train = train_emb_data_obj[sorted(potential_keys_train)[0]]

            elif train_emb_data_obj is not None:
                embedding_for_pik_train = train_emb_data_obj

            if train_is_correct_derived is not None and embedding_for_pik_train is not None:
                train_is_true.append(train_is_correct_derived)
                train_answerable.append(is_answerable(train_example))
                try:
                    if isinstance(embedding_for_pik_train, torch.Tensor):
                        train_embeddings.append(embedding_for_pik_train.cpu().flatten())
                    else:
                        train_embeddings.append(torch.tensor(embedding_for_pik_train).cpu().flatten())
                except Exception as e_conv_pik_train:
                     logging.warning(f"Could not convert/append embedding to tensor for P(IK) Train for ID {tid_train}: {e_conv_pik_train}")
            else:
                logging.warning(f"Skipping P(IK) training sample {tid_train}, missing accuracy label or suitable embedding.")

        train_is_false = [not b for b in train_is_true]
        train_unanswerable = [not b for b in train_answerable]

        if train_embeddings and validation_embeddings:
            target_len = len(validation_embeddings)
            if len(result_dict["validation_is_false"]) != target_len:
                 logging.warning(f"Padding/truncating validation_is_false for P(IK). Expected {target_len}, got {len(result_dict['validation_is_false'])}.")
                 result_dict["validation_is_false"] = (result_dict["validation_is_false"] + [False] * target_len)[:target_len]
            if len(result_dict["validation_unanswerable"]) != target_len:
                 logging.warning(f"Padding/truncating validation_unanswerable for P(IK). Expected {target_len}, got {len(result_dict['validation_unanswerable'])}.")
                 result_dict["validation_unanswerable"] = (result_dict["validation_unanswerable"] + [False] * target_len)[:target_len]

            if args.compute_p_ik:
                logging.info("Training P(IK) for correctness...")
                p_ik_preds = get_p_ik(
                    train_embeddings=train_embeddings, is_false=train_is_false,
                    eval_embeddings=validation_embeddings, eval_is_false=result_dict["validation_is_false"]
                )
                result_dict["uncertainty_measures"]["p_ik"] = p_ik_preds.tolist() if p_ik_preds is not None else [np.nan] * target_len
            if args.compute_p_ik_answerable:
                logging.info("Training P(IK) for answerability...")
                p_ik_ans_preds = get_p_ik(
                    train_embeddings=train_embeddings, is_false=train_unanswerable,
                    eval_embeddings=validation_embeddings, eval_is_false=result_dict["validation_unanswerable"]
                )
                result_dict["uncertainty_measures"]["p_ik_unanswerable"] = p_ik_ans_preds.tolist() if p_ik_ans_preds is not None else [np.nan] * target_len
        else:
            logging.warning("Skipping P(IK) calculation due to missing train or validation embeddings.")

    output_path = run_dir / OUTPUT_UNCERTAINTY_FILE
    try:
        final_uncertainty_measures = {}
        expected_len = count
        for key, score_list in result_dict["uncertainty_measures"].items():
            if len(score_list) == expected_len:
                final_uncertainty_measures[key] = score_list
            else:
                 logging.warning(f"Length mismatch for final score '{key}'. Expected {expected_len}, got {len(score_list)}. Padding/truncating.")
                 final_uncertainty_measures[key] = (score_list + [np.nan] * expected_len)[:expected_len]

        result_dict["uncertainty_measures"] = final_uncertainty_measures

        if "semantic_ids" in result_dict:
             if len(result_dict["semantic_ids"]) != expected_len:
                  logging.warning(f"Length mismatch for 'semantic_ids'. Expected {expected_len}, got {len(result_dict['semantic_ids'])}. Padding/truncating.")
                  result_dict["semantic_ids"] = (result_dict["semantic_ids"] + [[]] * expected_len)[:expected_len]

        with open(output_path, "wb") as f:
            pickle.dump(result_dict, f)
        logging.info(f"Final uncertainty measures (including NSE) saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save final uncertainty measures to {output_path}: {e}", exc_info=True)

    if entailment_model is not None and hasattr(entailment_model, 'model') and hasattr(entailment_model.model, 'to'):
        try:
            del entailment_model.model
            del entailment_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Cleaned up entailment model resources.")
        except Exception as cleanup_e:
            logging.warning(f"Could not fully clean up entailment model: {cleanup_e}")

    logging.info(f"--- SE/NSE Computation Stage Complete for Run: {run_id} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Semantic Entropy and other uncertainty measures.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing run files (pkls).")
    parser.add_argument("--entailment_model", type=str, default="deberta")
    parser.add_argument("--strict_entailment", action="store_true")
    parser.add_argument("--no-strict_entailment", dest="strict_entailment", action="store_false")
    parser.set_defaults(strict_entailment=True)
    parser.add_argument("--condition_on_question", action="store_true")
    parser.add_argument("--no-condition_on_question", dest="condition_on_question", action="store_false")
    parser.set_defaults(condition_on_question=True)
    parser.add_argument("--use_num_generations", type=int, default=-1)
    parser.add_argument("--use_all_generations", dest="use_all_generations_se", action="store_true")
    parser.add_argument("--no-use_all_generations", dest="use_all_generations_se", action="store_false")
    parser.set_defaults(use_all_generations_se=True)
    parser.add_argument("--num_eval_samples", type=int, default=int(1e19))
    parser.add_argument("--compute_predictive_entropy", action="store_true")
    parser.add_argument("--no-compute_predictive_entropy", dest="compute_predictive_entropy", action="store_false")
    parser.set_defaults(compute_predictive_entropy=True)
    parser.add_argument("--compute_p_ik", action="store_true", default=False)
    parser.add_argument("--compute_p_ik_answerable", action="store_true", default=False)
    parser.add_argument("--recompute_accuracy", action="store_true", default=False)
    parser.add_argument("--metric", default="squad", help="Metric for accuracy calculation")
    parser.add_argument("--metric_threshold", type=float, required=True, help="Metric threshold defining correctness.")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--entailment_cache_id", default=None)
    parser.add_argument("--entailment_cache_only", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Ignoring unknown args passed to compute_se.py: {unknown}")
    logging.info(f"Args received by compute_se.py: {args}")

    main(args)