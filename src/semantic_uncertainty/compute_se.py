from collections import defaultdict
import logging
import os
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch

from src.utils.p_ik import get_p_ik
from src.utils.semantic_entropy import (
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao,
    cluster_assignment_entropy,
    EntailmentDeberta,
    EntailmentLlama
)
from src.utils import utils
from src.utils import logging_utils

DEFAULT_VALIDATION_GEN_FILE = "validation_generations.pkl"
DEFAULT_TRAIN_GEN_FILE = "train_generations.pkl"
DEFAULT_OUTPUT_UNCERTAINTY_FILE = "uncertainty_measures.pkl"

def main(args):
    """
    Compute and save uncertainty measures for a given run, including semantic
    entropy, predictive entropy, cluster assignment entropy, and p(ik) values.

    Args:
        args: The parsed arguments from the command line.

    Returns:
        None
    """
    logging_utils.setup_logger()
    run_dir = Path(args.run_dir)
    run_name_for_log = run_dir.name
    is_synthetic_run_mode = bool(args.input_generations_filename_override)

    if args.input_generations_filename_override:
        run_name_for_log = f"{run_dir.name}_with_{Path(args.input_generations_filename_override).stem}"
    logging.info(f"--- Starting SE Computation for Run: {run_name_for_log} (dir: {run_dir}) ---")
    logging.info(f"Full args received by compute_se: {args}")
    if is_synthetic_run_mode:
        logging.info("Detected synthetic run mode based on filename override.")

    validation_gen_fname = args.input_generations_filename_override if args.input_generations_filename_override else DEFAULT_VALIDATION_GEN_FILE
    output_uncertainty_fname = args.output_uncertainty_filename_override if args.output_uncertainty_filename_override else DEFAULT_OUTPUT_UNCERTAINTY_FILE

    validation_generations_path = run_dir / validation_gen_fname
    validation_generations = utils.load_pickle(validation_generations_path)
    if validation_generations is None:
        logging.error(f"Failed to load {validation_generations_path}. Cannot proceed.")
        return

    result_dict = {}
    existing_uncertainty_output_path = run_dir / output_uncertainty_fname
    if existing_uncertainty_output_path.exists():
        logging.info(f"Loading existing uncertainty measures from: {existing_uncertainty_output_path}")
        loaded_data = utils.load_pickle(existing_uncertainty_output_path)
        if loaded_data: result_dict = loaded_data
        else: logging.warning(f"Failed to load existing {existing_uncertainty_output_path}. Initializing empty.")
    else:
        logging.info(f"No existing uncertainty file at {existing_uncertainty_output_path}. Will create a new one.")

    if "uncertainty_measures" not in result_dict or not isinstance(result_dict["uncertainty_measures"], dict):
        result_dict["uncertainty_measures"] = {}
    result_dict["uncertainty_measures"] = defaultdict(list, result_dict["uncertainty_measures"])
    
    calculated_entropies_this_run = defaultdict(list)
    calculated_semantic_ids_this_run = []
    
    ground_truth_is_true_this_run = []
    ground_truth_is_answerable_this_run = []
    eval_embeddings_for_pik_this_run = []
    processed_task_ids_this_run = []


    train_generations = None
    compute_pik_flag = args.compute_p_ik
    compute_pik_answerable_flag = args.compute_p_ik_answerable
    if compute_pik_flag or compute_pik_answerable_flag:
        train_gen_fname_to_load = args.input_train_generations_filename_override if args.input_train_generations_filename_override else DEFAULT_TRAIN_GEN_FILE
        train_generations_path = run_dir / train_gen_fname_to_load
        if train_generations_path.exists():
            train_generations = utils.load_pickle(train_generations_path)
            if train_generations is None: compute_pik_flag = False; compute_pik_answerable_flag = False
        else: compute_pik_flag = False; compute_pik_answerable_flag = False


    entailment_model = None
    if args.compute_predictive_entropy:
        logging.info("Loading entailment model: %s", args.entailment_model)
        try:
            if ("BiomedNLP-PubMedBERT" in args.entailment_model or args.entailment_model == "deberta"):
                entailment_model_name = args.entailment_model if args.entailment_model != "deberta" else "lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli"
                entailment_model = EntailmentDeberta(model_name=entailment_model_name)
            elif "llama" in args.entailment_model.lower():
                 entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
            else: logging.error(f"Unsupported entailment model type: {args.entailment_model}"); return
            if entailment_model: logging.info("Entailment model loading complete.")
        except Exception as e: logging.error(f"Failed to load entailment model '{args.entailment_model}': {e}", exc_info=True); return
    else: logging.info("compute_predictive_entropy is False. Skipping SE/NSE/RE/CAE calculations.")


    samples_to_process_ids = list(validation_generations.keys())
    num_eval_samples_limit = args.num_eval_samples if hasattr(args, 'num_eval_samples') and args.num_eval_samples is not None else len(samples_to_process_ids)
    
    metric_for_recompute_accuracy = None
    if args.recompute_accuracy:
        logging.info(f"Accuracy recomputation enabled with metric: {args.metric}")
        metric_for_recompute_accuracy = utils.get_metric(args.metric)

    logging.info(f"Starting loop for {min(num_eval_samples_limit, len(samples_to_process_ids))} validation samples...")

    def is_answerable_local_util(gen_ex):
        """
        Local utility function to determine if a generation example is answerable
        (i.e., has a reference answer).

        Args:
            gen_ex (dict): Generation example dictionary with a "reference" key
                containing an "answers" key with a "text" key.

        Returns:
            bool: Whether the example is answerable (i.e., has a reference answer).

        """
        ref = gen_ex.get("reference", {}); ans = ref.get("answers", {}); txt = ans.get("text", []); return bool(txt)

    for idx, tid in enumerate(samples_to_process_ids):
        if idx >= num_eval_samples_limit: break
        processed_task_ids_this_run.append(tid)

        example = validation_generations[tid]
        most_likely_answer = example.get("most_likely_answer", {})
        
        current_accuracy_value = most_likely_answer.get("accuracy")
        if args.recompute_accuracy and metric_for_recompute_accuracy:
            response_text = most_likely_answer.get("response")
            if response_text and response_text != "[PREDICTION FAILED]":
                try: current_accuracy_value = metric_for_recompute_accuracy(response_text, example, None)
                except Exception as e_metric: logging.warning(f"Metric recalc failed for {tid}: {e_metric}"); current_accuracy_value = 0.0
            else: current_accuracy_value = 0.0
        elif current_accuracy_value is None:
            logging.warning(f"Accuracy missing for ID {tid}. Assuming incorrect for this run's ground truth.")
            current_accuracy_value = 0.0
            
        is_correct_flag = (current_accuracy_value > args.probe_accuracy_threshold)
        ground_truth_is_true_this_run.append(is_correct_flag)
        ground_truth_is_answerable_this_run.append(is_answerable_local_util(example))

        if compute_pik_flag or compute_pik_answerable_flag:
            embedding = most_likely_answer.get("embedding")
            if embedding is not None:
                try:
                    if isinstance(embedding, torch.Tensor): eval_embeddings_for_pik_this_run.append(embedding.cpu())
                    else: eval_embeddings_for_pik_this_run.append(torch.tensor(embedding).cpu())
                except:
                    pass
            elif compute_pik_flag:
                pass


        if args.compute_predictive_entropy and entailment_model:
            full_responses = example.get("responses", [])
            log_liks = [r[1] for r in full_responses if isinstance(r, (list, tuple)) and len(r) > 1 and r[1] is not None]
            responses = [r[0] for r in full_responses if isinstance(r, (list, tuple)) and len(r) > 0]
            se_val, nse_val, re_val, cae_val, s_ids_val = np.nan, np.nan, np.nan, np.nan, []
            if log_liks and responses and len(log_liks) == len(responses):
                try:
                    s_ids_val = get_semantic_ids(responses,model=entailment_model,strict_entailment=args.strict_entailment,example=example)
                    log_liks_agg = [np.mean(ll) if ll and all(isinstance(x, (int, float)) for x in ll) else np.nan for ll in log_liks]
                    if not np.isnan(log_liks_agg).any():
                        cae_val = cluster_assignment_entropy(s_ids_val)
                        re_val = predictive_entropy_rao(log_liks_agg)
                        log_lik_per_id = logsumexp_by_id(s_ids_val, log_liks_agg, agg="sum_normalized")
                        n_clusters = len(log_lik_per_id)
                        if n_clusters > 0:
                            se_val = predictive_entropy_rao(log_lik_per_id)
                            if n_clusters <=1: 
                                nse_val = 0.0
                            else:
                                max_e = np.log2(n_clusters)
                                if max_e > 1e-9 and not np.isnan(se_val): nse_val = se_val / max_e
                except Exception as e: logging.error(f"Error calculating entropies for {tid}: {e}", exc_info=True)
            calculated_entropies_this_run["semantic_entropy"].append(se_val)
            calculated_entropies_this_run["normalized_semantic_entropy"].append(nse_val)
            calculated_entropies_this_run["regular_entropy"].append(re_val)
            calculated_entropies_this_run["cluster_assignment_entropy"].append(cae_val)
            calculated_semantic_ids_this_run.append(s_ids_val)

    logging.info(f"Finished processing loop. Samples processed in this execution: {len(processed_task_ids_this_run)}")

    result_dict["task_ids"] = processed_task_ids_this_run
    result_dict["validation_is_false"] = [not b for b in ground_truth_is_true_this_run]
    result_dict["validation_unanswerable"] = [not b for b in ground_truth_is_answerable_this_run]
    
    if args.compute_predictive_entropy:
        result_dict["semantic_ids"] = calculated_semantic_ids_this_run
        for key, value_list in calculated_entropies_this_run.items():
            result_dict["uncertainty_measures"][key] = value_list
    
    if (compute_pik_flag or compute_pik_answerable_flag) and train_generations:
        train_is_true_pik, train_embeddings_pik, train_answerable_pik = [], [], []
        if train_generations:
            for tid_train, train_example in train_generations.items():
                train_most_likely = train_example.get("most_likely_answer", {})
                train_acc = train_most_likely.get("accuracy")
                train_emb = train_most_likely.get("embedding")
                if train_acc is not None and train_emb is not None:
                    train_is_true_pik.append(train_acc > args.probe_accuracy_threshold)
                    train_answerable_pik.append(is_answerable_local_util(train_example))
                    try:
                        if isinstance(train_emb, torch.Tensor): train_embeddings_pik.append(train_emb.cpu())
                        else: train_embeddings_pik.append(torch.tensor(train_emb).cpu())
                    except: pass
                else: pass
        train_is_false_pik = [not b for b in train_is_true_pik]
        train_unanswerable_pik = [not b for b in train_answerable_pik]

        if train_embeddings_pik and eval_embeddings_for_pik_this_run:
            aligned_eval_is_false = result_dict["validation_is_false"][:len(eval_embeddings_for_pik_this_run)]
            aligned_eval_unanswerable = result_dict["validation_unanswerable"][:len(eval_embeddings_for_pik_this_run)]

            if compute_pik_flag:
                p_ik_preds = get_p_ik(train_embeddings_pik, train_is_false_pik, eval_embeddings_for_pik_this_run, aligned_eval_is_false)
                result_dict["uncertainty_measures"]["p_ik"] = p_ik_preds.tolist() if p_ik_preds is not None else [np.nan] * len(aligned_eval_is_false)
            if compute_pik_answerable_flag:
                p_ik_ans_preds = get_p_ik(train_embeddings_pik, train_unanswerable_pik, eval_embeddings_for_pik_this_run, aligned_eval_unanswerable)
                result_dict["uncertainty_measures"]["p_ik_unanswerable"] = p_ik_ans_preds.tolist() if p_ik_ans_preds is not None else [np.nan] * len(aligned_eval_unanswerable)
        else:
            if compute_pik_flag: result_dict["uncertainty_measures"]["p_ik"] = [np.nan] * len(processed_task_ids_this_run)
            if compute_pik_answerable_flag: result_dict["uncertainty_measures"]["p_ik_unanswerable"] = [np.nan] * len(processed_task_ids_this_run)

    final_len = len(processed_task_ids_this_run)
    for key in list(result_dict["uncertainty_measures"].keys()):
        current_list = result_dict["uncertainty_measures"][key]
        if len(current_list) != final_len:
            padded_list = (list(current_list) + [np.nan] * final_len)[:final_len]
            result_dict["uncertainty_measures"][key] = padded_list
            
    output_path = run_dir / output_uncertainty_fname
    try:
        with open(output_path, "wb") as f:
            pickle.dump(result_dict, f)
        logging.info(f"Final uncertainty measures saved to: {output_path}")
        logging.info(f"Keys in saved uncertainty_measures: {list(result_dict['uncertainty_measures'].keys())}")
        if "validation_is_false" in result_dict:
            logging.info(f"Length of validation_is_false: {len(result_dict['validation_is_false'])}")
        if "task_ids" in result_dict:
            logging.info(f"Length of task_ids: {len(result_dict['task_ids'])}")
    except Exception as e:
        logging.error(f"Failed to save final data to {output_path}: {e}", exc_info=True)

    if entailment_model is not None and hasattr(entailment_model, 'model'):
        try:
            del entailment_model.model
            del entailment_model
            torch.cuda.empty_cache()
        except: pass
    logging.info(f"--- SE Computation Stage Complete for: {run_name_for_log} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Semantic Entropy and other uncertainty measures.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--entailment_model", type=str, default="deberta")
    parser.add_argument("--strict_entailment", action="store_true") 
    parser.add_argument("--no-strict_entailment", dest="strict_entailment", action="store_false")
    parser.set_defaults(strict_entailment=True) 
    parser.add_argument("--condition_on_question", action="store_true") 
    parser.add_argument("--no-condition_on_question", dest="condition_on_question", action="store_false")
    parser.set_defaults(condition_on_question=True) 
    parser.add_argument("--use_num_generations", type=int, default=-1) 
    parser.add_argument("--use_all_generations_se", dest="use_all_generations_se", action="store_true")
    parser.add_argument("--no-use_all_generations_se", dest="use_all_generations_se", action="store_false")
    parser.set_defaults(use_all_generations_se=True) 
    parser.add_argument("--num_eval_samples", type=int, default=int(1e19)) 
    parser.add_argument("--compute_predictive_entropy", action="store_true")
    parser.add_argument("--no-compute_predictive_entropy", dest="compute_predictive_entropy", action="store_false")
    parser.set_defaults(compute_predictive_entropy=True) 
    parser.add_argument("--compute_p_ik", action="store_true", default=False)
    parser.add_argument("--compute_p_ik_answerable", action="store_true", default=False)
    parser.add_argument("--probe_accuracy_threshold", type=float, default=0.5)
    parser.add_argument("--entailment_cache_id", default=None)
    parser.add_argument("--entailment_cache_only", action="store_true", default=False)
    parser.add_argument("--recompute_accuracy", action="store_true", default=False) 
    parser.add_argument("--metric", default="squad") 

    parser.add_argument("--input_generations_filename_override", default=None)
    parser.add_argument("--input_train_generations_filename_override", default=None)
    parser.add_argument("--output_uncertainty_filename_override", default=None)
    
    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Ignoring unknown args in compute_se: {unknown}")
    main(args)