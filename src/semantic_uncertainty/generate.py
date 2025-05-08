import gc
import os
import logging
import random
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch
import wandb

from src.utils.data_utils import load_ds
from src.utils import utils
from src.utils import p_true as p_true_utils
from src.utils import logging_utils

from .compute_se import main as main_compute

def generate_answers(args):
    logging_utils.setup_logger()
    logging.info(f"Starting answer generation with args: {args}")

    run_output_dir = Path(args.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_output_dir.name

    experiment_details = {"args": args.__dict__}
    random.seed(args.random_seed)

    annotation_output_file = (
        run_output_dir / f"{run_id}_validation_outputs_for_annotation.jsonl"
    )
    logging.info(f"Annotation outputs jsonl will be saved to: {annotation_output_file}")

    metric = utils.get_metric(args.metric)

    ds_options = {}
    if args.dataset.lower() == "medquad" and args.summarize_medquad:
        ds_options["summarize_answers"] = True
    if args.num_samples:
        ds_options["num_samples"] = args.num_samples

    logging.info(f"Loading dataset: {args.dataset} with options: {ds_options}")
    train_dataset, validation_dataset = load_ds(
        args.dataset,
        seed=args.random_seed,
        add_options=ds_options if ds_options else None,
    )
    if train_dataset is None:
        logging.error("Failed to load datasets. Exiting generation.")
        return

    logging.info("Splitting dataset indices...")
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if not answerable_indices:
        logging.error(
            "No answerable indices found in the training set. Cannot create few-shot prompts."
        )
        return

    num_few_shot = min(args.num_few_shot, len(answerable_indices))
    if num_few_shot < args.num_few_shot:
        logging.warning(
            f"Requested {args.num_few_shot} few-shot examples, but only {len(answerable_indices)} answerable indices available. Using {num_few_shot}."
        )

    prompt_indices = random.sample(answerable_indices, num_few_shot)
    experiment_details["prompt_indices"] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS.get(args.brief_prompt, utils.BRIEF_PROMPTS["default"])
    is_brief_always = args.brief_always if args.enable_brief else True

    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, is_brief_always, make_prompt
    )
    experiment_details["prompt"] = prompt
    experiment_details["BRIEF"] = BRIEF
    logging.info("Few-shot prompt constructed.")

    logging.info("Initializing model...")
    model = utils.init_model(args)
    if model is None:
        logging.error("Failed to initialize model.")
        return

    p_true_few_shot_prompt = None
    p_true_responses_for_prompt_construction = {} # Store responses used ONLY for prompt construction if needed later
    if args.compute_p_true:
        logging.info("Constructing few-shot prompt for P(True)...")
        available_for_ptrue = list(set(remaining_answerable))
        if not available_for_ptrue:
            logging.warning(
                "No remaining answerable indices for P(True) few-shot prompt. Skipping P(True) calculation."
            )
            args.compute_p_true = False
        else:
            num_ptrue_fewshot = min(args.p_true_num_fewshot, len(available_for_ptrue))
            if num_ptrue_fewshot < args.p_true_num_fewshot:
                logging.warning(
                    f"Requested {args.p_true_num_fewshot} P(True) few-shot examples, but only {len(available_for_ptrue)} remaining answerable indices available. Using {num_ptrue_fewshot}."
                )

            p_true_indices = random.sample(available_for_ptrue, num_ptrue_fewshot)
            remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

            p_true_few_shot_prompt, p_true_responses_for_prompt_construction, len_p_true = (
                p_true_utils.construct_few_shot_prompt(
                    model=model,
                    dataset=train_dataset,
                    indices=p_true_indices,
                    prompt=prompt,
                    brief=BRIEF,
                    brief_always=is_brief_always,
                    make_prompt=make_prompt,
                    num_generations=args.num_generations,
                    metric=lambda pred, ex, mdl: utils.get_metric(args.metric)(pred, ex, mdl) > args.metric_threshold, # Pass a boolean lambda
                    top_p=args.top_p,
                )
            )
            experiment_details["p_true_indices"] = p_true_indices
            # experiment_details["p_true_responses"] = p_true_responses_for_prompt_construction # Optional: only if needed for debug
            experiment_details["p_true_few_shot_prompt"] = p_true_few_shot_prompt
            logging.info("Finished constructing P(True) few-shot prompt.")

    logging.info("=" * 80)
    logging.info("Generating answers...")
    logging.info("=" * 80)

    for dataset_split in ["train", "validation"]:
        logging.info(f"\n--- Processing split: {dataset_split} ---")

        if dataset_split == "train" and not args.get_training_set_generations:
            logging.info("Skipping training data generation as per config.")
            continue

        current_dataset = (
            train_dataset if dataset_split == "train" else validation_dataset
        )
        possible_indices = []
        if dataset_split == "train":
            possible_indices = list(
                set(remaining_answerable) | set(unanswerable_indices)
            )
        else:
            possible_indices = list(range(len(current_dataset)))

        if args.answerable_only and dataset_split == "validation":
            logging.info("Filtering validation set for answerable questions only.")
            # Re-filter the actual validation dataset object being used
            current_dataset = [
                ds for ds in current_dataset if len(ds["answers"]["text"]) > 0
            ]
            possible_indices = list(range(len(current_dataset))) # Update indices based on filtered list


        if not possible_indices:
            logging.warning(
                f"No indices to process for split: {dataset_split}. Skipping."
            )
            continue

        num_to_process = min(args.num_samples, len(possible_indices))
        if num_to_process < args.num_samples:
            logging.warning(
                f"Requested {args.num_samples} samples for {dataset_split}, but only {len(possible_indices)} available. Processing {num_to_process}."
            )

        indices_to_process = random.sample(possible_indices, num_to_process)
        experiment_details[dataset_split] = {"indices": indices_to_process}

        generations = {}
        accuracies = [] # Store final binary accuracy for the split average log

        annotation_file_handle = None
        if dataset_split == "validation":
            try:
                annotation_file_handle = open(
                    annotation_output_file, "w", encoding="utf-8"
                )
                logging.info(f"Opened annotation output file: {annotation_output_file}")
            except IOError as e:
                logging.error(
                    f"Could not open annotation file {annotation_output_file} for writing: {e}"
                )
                annotation_file_handle = None

        for it, index in enumerate(
            tqdm(indices_to_process, desc=f"Generating {dataset_split}")
        ):
            if (it + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            example = current_dataset[index]
            question = example.get("question", "")
            context = example.get("context", "")
            example_id = example.get("id", f"{dataset_split}_{index}")
            correct_answers_list = example.get("answers", {}).get("text", [])

            generations[example_id] = {"question": question, "context": context}

            current_input = make_prompt(context, question, None, BRIEF, is_brief_always)
            local_prompt = prompt + current_input

            full_responses = []
            most_likely_answer_dict_anno = {}

            is_train_most_likely_only = (
                dataset_split == "train"
                and args.get_training_set_generations
                and args.get_training_set_generations_most_likely_only
            )
            num_loops = 1 if is_train_most_likely_only else args.num_generations + 1

            for i in range(num_loops):
                temperature = 0.1 if i == 0 else args.temperature
                top_p_val = args.top_p

                try:
                    predicted_answer, token_log_likelihoods, multi_layer_embeddings_dict = model.predict(
                        local_prompt, temperature, top_p_val
                    )

                except Exception as e:
                    logging.error(
                        f"Model prediction failed for example {example_id}, iteration {i}: {e}"
                    )
                    predicted_answer = "[PREDICTION FAILED]"
                    token_log_likelihoods = []
                    multi_layer_embeddings_dict = None

                loop_type = (
                    "Low-T" if i == 0 else f"High-T ({i}/{args.num_generations})"
                )
                print(
                    f"  ID {example_id} [{loop_type}]: Generated='{predicted_answer}'",
                    flush=True,
                )

                metric_score = 0.0
                is_correct_label = False
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if (
                    correct_answers_list
                    and compute_acc
                    and predicted_answer != "[PREDICTION FAILED]"
                ):
                    try:
                        metric_score = metric(predicted_answer, example, model)
                        is_correct_label = metric_score > args.metric_threshold
                    except Exception as e:
                        logging.error(
                            f"Metric calculation failed for ID {example_id}, iter {i}: {e}"
                        )
                        metric_score = 0.0
                        is_correct_label = False

                if i == 0:
                    logging.info(
                        f"ID {example_id}: Low-T Gen='{predicted_answer[:60]}...', {args.metric}_Score={metric_score:.4f}, Correct(>{args.metric_threshold})={is_correct_label}"
                    )
                    accuracies.append(1.0 if is_correct_label else 0.0)
                    most_likely_answer_dict_gen = {
                        "response": predicted_answer,
                        "token_log_likelihoods": token_log_likelihoods,
                        "embedding": multi_layer_embeddings_dict,
                        "accuracy_metric_score": float(metric_score),
                        "is_correct": is_correct_label
                    }
                    generations[example_id][
                        "most_likely_answer"
                    ] = most_likely_answer_dict_gen
                    generations[example_id]["reference"] = utils.get_reference(example)

                    most_likely_answer_dict_anno = {
                        "response": predicted_answer,
                        "accuracy_score": float(metric_score),
                        "is_correct": is_correct_label,
                    }

                else:
                     high_t_metric_score = 0.0
                     high_t_is_correct = False
                     if correct_answers_list and args.compute_accuracy_at_all_temps and predicted_answer != "[PREDICTION FAILED]":
                         try:
                            high_t_metric_score = metric(predicted_answer, example, model)
                            high_t_is_correct = high_t_metric_score > args.metric_threshold
                         except Exception:
                             high_t_metric_score = 0.0; high_t_is_correct=False

                     full_responses.append(
                        (predicted_answer, token_log_likelihoods, multi_layer_embeddings_dict, float(high_t_metric_score))
                    )

            generations[example_id]["responses"] = full_responses

            if (
                args.compute_p_true
                and p_true_few_shot_prompt is not None
                and dataset_split == "validation"
                and "most_likely_answer" in generations[example_id] # Make sure low-T response exists
            ):
                 p_true_score = p_true_utils.calculate_p_true(
                     model,
                     question,
                     generations[example_id]["most_likely_answer"]["response"],
                     [r[0] for r in full_responses],
                     p_true_few_shot_prompt,
                     hint=args.p_true_hint,
                 )
                 # Store directly in the generations dict for this sample
                 generations[example_id]['p_true_logprob'] = p_true_score
            elif dataset_split == "validation":
                 generations[example_id]['p_true_logprob'] = np.nan # Ensure field exists even if not computed

            if dataset_split == "validation" and annotation_file_handle:
                annotation_record = {
                    "id": example_id,
                    "question": question,
                    "context": (context if context is not None else ""),
                    "reference_answers": correct_answers_list,
                    "generated_answer": most_likely_answer_dict_anno.get(
                        "response", "[GENERATION FAILED]"
                    ),
                    "is_correct": most_likely_answer_dict_anno.get("is_correct", False),
                    "accuracy_metric_score": most_likely_answer_dict_anno.get(
                        "accuracy_score", 0.0
                    ),
                    "hallucination_type": "",
                    "hallucination_subtype": "",
                    "annotation_notes": "",
                    "high_temp_samples": [resp[0] for resp in full_responses],
                }
                try:
                    annotation_file_handle.write(json.dumps(annotation_record) + "\n")
                except Exception as e:
                    logging.error(
                        f"Failed to write annotation record for {example_id} to file: {e}"
                    )

        if annotation_file_handle:
            annotation_file_handle.close()
            logging.info(f"Closed annotation output file: {annotation_output_file}")

        pkl_filename = f"{dataset_split}_generations.pkl"
        pkl_filepath = run_output_dir / pkl_filename
        try:
            with open(pkl_filepath, "wb") as f:
                pickle.dump(generations, f)
            logging.info(f"Saved generations pickle to {pkl_filepath}")
        except Exception as e:
            logging.error(f"Failed to save pickle file {pkl_filepath}: {e}")

        if accuracies:
            overall_accuracy = np.mean(accuracies)
            logging.info(
                f"Overall {dataset_split} split accuracy: {overall_accuracy:.4f}"
            )
        else:
            logging.warning(f"No accuracy scores recorded for {dataset_split} split.")

    exp_details_filename = "experiment_details.pkl"
    exp_details_filepath = run_output_dir / exp_details_filename
    try:
        if isinstance(experiment_details.get("args"), argparse.Namespace):
            experiment_details["args"] = vars(experiment_details["args"])

        # Remove sensitive or large data not needed downstream from details
        # experiment_details.pop("p_true_responses", None)

        with open(exp_details_filepath, "wb") as f:
            pickle.dump(experiment_details, f)
        logging.info(f"Saved experiment details to {exp_details_filepath}")
    except Exception as e:
        logging.error(f"Failed to save experiment details {exp_details_filepath}: {e}")

    logging.info(f"Generation stage complete for run {run_id}.")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers using specified LLM and config."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save run outputs."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--num_few_shot", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--enable_brief", action="store_true")
    parser.add_argument("--brief_prompt", default="default")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--metric", default="squad")
    parser.add_argument(
        "--metric_threshold",
        type=float,
        required=True,
        help="Threshold for the chosen metric (e.g., SQuAD F1, BERTScore F1) to define correctness.",
    )
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--compute_p_true", action="store_true")
    parser.add_argument("--p_true_num_fewshot", type=int, default=20)
    parser.add_argument("--model_max_new_tokens", type=int, default=50)
    parser.add_argument("--brief_always", action="store_true")
    parser.add_argument("--prompt_type", default="default")
    parser.add_argument("--answerable_only", action="store_true")
    parser.add_argument("--summarize_medquad", action="store_true")
    parser.add_argument(
        "--get_training_set_generations", action="store_true", default=True
    )
    parser.add_argument(
        "--get_training_set_generations_most_likely_only",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--compute_accuracy_at_all_temps", action="store_true", default=True
    )
    parser.add_argument("--p_true_hint", action="store_true", default=False)
    parser.add_argument("--entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--experiment_lot", type=str, default="generate_stage_run")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of high-temperature generations for SE/P(True) calculation.",
    )
    parser.add_argument(
        "--probe_layers_to_extract",
        type=int,
        nargs="+",
        default=[-1],
        help="List of layer indices (negative for from_end) to extract embeddings for IS probe.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Ignoring unknown arguments passed to generate.py: {unknown}")

    generate_answers(args)