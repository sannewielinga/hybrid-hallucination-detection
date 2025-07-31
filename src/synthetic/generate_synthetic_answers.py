import gc
import os
import logging
import random
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import sys
import re

project_root_path = Path(__file__).resolve().parent.parent.parent 
if str(project_root_path) not in sys.path:
    sys.path.insert(0, str(project_root_path))

from src.utils import utils as main_utils
from src.utils import p_true as p_true_utils
from src.utils import logging_utils

SYNTHETIC_DOSAGE_SYSTEM_PROMPT = "You are an expert at medical dosage calculations. Respond with ONLY the final numerical dosage value. Do NOT include units. Do NOT include any other text, reasoning, or explanations. Just the number."

def extract_numerical_value(text_response):
    """
    Given a text response, try to extract a numerical value from it. Only matches the first
    numerical value found in the response. Returns None if no numerical value is found.

    Example input: "The patient should receive 10mg of the medication."
    Example output: "10"

    :param text_response: A string from an LLM which we want to extract a numerical value from.
    :return: The extracted numerical value, or None if no numerical value is found.
    """
    if text_response is None:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text_response)
    if match:
        return match.group(0)
    return None

def numerical_correctness_metric(predicted_answer_str, example_synthetic_prompt, model_unused):
    """
    Compute the correctness metric for a synthetic dosage prompt.

    :param predicted_answer_str: The text output from the LLM.
    :param example_synthetic_prompt: The synthetic dosage prompt example, which contains the ground truth dosage.
    :param model_unused: The model used to generate the answer (unused in this metric).

    :return: 1.0 if the predicted dosage matches the ground truth dosage (with a small tolerance), 0.0 otherwise.
    """
    extracted_number_str = extract_numerical_value(predicted_answer_str)
    if extracted_number_str is None:
        logging.warning(f"Could not extract numerical value from LLM output '{predicted_answer_str}' for ID {example_synthetic_prompt['id']}.")
        return 0.0
    try:
        predicted_dosage = float(extracted_number_str)
        ground_truth_dosage = float(example_synthetic_prompt["ground_truth_dosage"])
        return 1.0 if abs(predicted_dosage - ground_truth_dosage) < 1e-6 else 0.0
    except ValueError: 
        logging.warning(f"Could not convert extracted string '{extracted_number_str}' to float for ID {example_synthetic_prompt['id']}.")
        return 0.0
    except Exception as e:
        logging.error(f"Error in numerical_correctness_metric for ID {example_synthetic_prompt['id']}: {e}")
        return 0.0

def generate_synthetic_answers(args):
    """
    Generates synthetic answers for dosage prompts using a specified language model.

    This function performs the following tasks:
        - Sets up logging and initializes output directories.
        - Loads synthetic dosage prompts from the specified input file.
        - Initializes the model with given parameters.
        - Iterates through the prompts, generating responses using the model.
        - Computes accuracy of the generated answers against ground truth.
        - Stores generated answers, embeddings, and accuracy metrics.
        - Saves the results to a pickle file for later analysis.

    Args:
        args (argparse.Namespace): A namespace containing configuration parameters for
            synthetic answer generation, including model parameters and input/output paths.

    Returns:
        None
    """

    logging_utils.setup_logger()
    logging.info(f"Starting SYNTHETIC answer generation with args: {args}")

    run_output_dir = Path(args.output_dir_synthetic)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name_slug = args.model_name.replace('/', '_')
    input_prompts_stem = Path(args.input_synthetic_prompts).stem
    run_id_synthetic = f"synth_{model_name_slug}_{input_prompts_stem}"


    random.seed(args.random_seed)

    synthetic_prompts = []
    try:
        with open(args.input_synthetic_prompts, 'r') as f:
            for line in f:
                synthetic_prompts.append(json.loads(line))
        logging.info(f"Loaded {len(synthetic_prompts)} synthetic prompts from {args.input_synthetic_prompts}")
    except Exception as e:
        logging.error(f"Failed to load synthetic prompts: {e}"); return
    
    if not synthetic_prompts:
        logging.error("No synthetic prompts loaded. Exiting."); return

    metric = numerical_correctness_metric

    logging.info("Initializing model...")
    model_init_args = argparse.Namespace(
        model_name=args.model_name,
        base_model=args.base_model,
        model_max_new_tokens=args.model_max_new_tokens
    )
    model = main_utils.init_model(model_init_args)
    if model is None:
        logging.error("Failed to initialize model.")
        return

    logging.info("=" * 80)
    logging.info(f"Generating SYNTHETIC answers for {run_id_synthetic}...")
    logging.info("=" * 80)

    generations_data_synthetic = {}
    indices_to_process = list(range(len(synthetic_prompts)))
    if args.num_samples_synthetic is not None:
        indices_to_process = indices_to_process[:args.num_samples_synthetic]

    accuracies_synthetic = []

    for it, index in enumerate(tqdm(indices_to_process, desc=f"Generating for {run_id_synthetic}")):
        if (it + 1) % 20 == 0:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        example_synthetic = synthetic_prompts[index]
        example_id = example_synthetic["id"]
        llm_prompt_text = example_synthetic["prompt_text"]
        
        generations_data_synthetic[example_id] = {
            "question": llm_prompt_text,
            "context": None,
            "reference_answers": {"text": [str(example_synthetic["ground_truth_dosage"])], "answer_start": [0]},
            "ground_truth_dosage": example_synthetic["ground_truth_dosage"],
            "units": example_synthetic["units"],
            "medication_name": example_synthetic["medication_name"],
            "distractors_in_prompt": example_synthetic["distractors_in_prompt"]
        }

        full_responses_for_se = []
        num_loops = 1 if args.get_training_set_generations_most_likely_only else args.num_generations + 1

        for i in range(num_loops):
            temperature_to_use = 1.0
            top_p_to_use = 1.0
            do_sample_flag = False
            loop_type = "Low-T (Numerical - Greedy)"

            if i > 0:
                temperature_to_use = args.temperature_synthetic
                top_p_to_use = args.top_p
                do_sample_flag = True
                loop_type = f"High-T ({i}/{args.num_generations})"
            one_shot_example_for_user_prompt = (
                "Example Question:\n"
                "The dosing rule for DrugX is 5 mg/kg. A patient weighs 10 kg. Distractor: they have a blue hat. What is the dosage in mg? Respond with only the numerical value.\n"
                "Example Answer: 50\n\n"
                "Now, your turn:\n"
            )
            final_llm_input = f"<|system|>\n{SYNTHETIC_DOSAGE_SYSTEM_PROMPT}</s>\n<|user|>\n{one_shot_example_for_user_prompt}{llm_prompt_text}</s>\n<|assistant|>\n"
            
            try:
                predicted_answer_raw, token_log_likelihoods, embedding = model.predict(
                    final_llm_input, 
                    temperature_to_use,
                    top_p=top_p_to_use,
                    do_sample=do_sample_flag
                )
                if embedding is not None and not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding)
            except Exception as e:
                logging.error(f"Model prediction failed for synth_id {example_id}, iteration {i} ({loop_type}): {e}")
                predicted_answer_raw = "[PREDICTION FAILED]"
                token_log_likelihoods = []; embedding = None

            logging.debug(f"  Synth ID {example_id} [{loop_type}]: Raw Gen='{predicted_answer_raw}'")

            acc = 0.0
            if i == 0 and predicted_answer_raw != "[PREDICTION FAILED]":
                try:
                    acc = metric(predicted_answer_raw, example_synthetic, model)
                except Exception as e:
                    logging.error(f"Metric calculation failed for synth_id {example_id}, iter {i}: {e}")
                    acc = 0.0
            
            if i == 0:
                logging.info(f"Synth ID {example_id}: Raw Low-T Gen='{predicted_answer_raw.strip()}'")
                logging.info(f"Synth ID {example_id}: Processed Low-T, Acc={acc:.2f}")
                accuracies_synthetic.append(acc)
                
                final_response_to_store = extract_numerical_value(predicted_answer_raw.strip())
                if final_response_to_store is None:
                    final_response_to_store = predicted_answer_raw.strip()

                most_likely_answer_dict_gen = {
                    "response": final_response_to_store,
                    "token_log_likelihoods": token_log_likelihoods,
                    "embedding": embedding,
                    "accuracy": float(acc),
                }
                generations_data_synthetic[example_id]["most_likely_answer"] = most_likely_answer_dict_gen
            else:
                 full_responses_for_se.append(
                    (predicted_answer_raw.strip(), token_log_likelihoods, embedding, 0.0)
                )
        
        generations_data_synthetic[example_id]["responses"] = full_responses_for_se

    output_filename = f"{run_id_synthetic}_generations_synthetic_dosage.pkl"
    output_filepath = run_output_dir / output_filename
    try:
        with open(output_filepath, "wb") as f:
            pickle.dump(generations_data_synthetic, f)
        logging.info(f"Saved synthetic generations pickle to {output_filepath}")
    except Exception as e:
        logging.error(f"Failed to save synthetic generations pickle {output_filepath}: {e}")

    if accuracies_synthetic:
        overall_accuracy = np.mean(accuracies_synthetic)
        logging.info(f"Overall SYNTHETIC dosage accuracy for {run_id_synthetic}: {overall_accuracy:.4f}")
    else:
        logging.warning(f"No accuracy scores recorded for synthetic run {run_id_synthetic}.")

    logging.info(f"Synthetic answer generation stage complete for {run_id_synthetic}.")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM answers for synthetic dosage prompts.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--model_max_new_tokens", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--num_generations", type=int, default=10)
    parser.add_argument("--temperature_synthetic", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--input_synthetic_prompts", type=str, required=True)
    parser.add_argument("--output_dir_synthetic", type=str, required=True)
    parser.add_argument("--num_samples_synthetic", type=int, default=None)
    parser.add_argument("--get_training_set_generations_most_likely_only", action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    if unknown:
        logging.warning(f"Ignoring unknown args: {unknown}")
    
    generate_synthetic_answers(args)