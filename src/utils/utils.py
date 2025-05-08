# src/utils/utils.py

import os
import logging
import argparse
import pickle
from pathlib import Path
import copy
import torch
import accelerate
import wandb
import numpy as np
import pandas as pd

from collections import Counter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, PeftConfig
from evaluate import load
from abc import ABC, abstractmethod
from typing import List, Text

from huggingface_hub import snapshot_download


BRIEF_PROMPTS = {
    "default": "Answer the following question as briefly as possible.\n",
    "chat": "Answer the following question in a single brief but complete sentence.\n",
}

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]

class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == "tokens":
            self.stops = [
                torch.tensor(self.tokenizer.encode(i)).to("cuda") for i in self.stops
            ]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == "text":
                generation = self.tokenizer.decode(
                    input_ids[0][self.initial_length :], skip_special_tokens=False
                )
                match = stop in generation
            elif self.match_on == "tokens":
                match = stop in input_ids[0][-len(stop) :]
            else:
                raise
            if match:
                return True
        return False

def remove_split_layer(device_map_in):
    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter([".".join(i.split(".")[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            raise ValueError(
                "More than one split layer.\n"
                f"Currently at layer {layer}.\n"
                f"In map: {device_map_in}\n"
                f"Out map: {device_map}\n"
            )

        logging.info(f"Split layer is {layer}.")

        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f"pop {name}")
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map

class HuggingfaceModel(BaseModel):

    def __init__(
        self, model_name, stop_sequences=None, max_new_tokens=None, base_model=None,
        probe_layers_to_extract=None
    ):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be provided.")
        self.max_new_tokens = max_new_tokens
        self.probe_layers_to_extract = probe_layers_to_extract if probe_layers_to_extract is not None else [-1]
        logging.info(f"HuggingfaceModel configured to extract probe embeddings from layers: {self.probe_layers_to_extract}")

        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        is_peft_lora = False
        try:
            _ = PeftConfig.from_pretrained(model_name)
            is_adapter_checkpoint = True
        except Exception as e:
            is_adapter_checkpoint = False

        if is_adapter_checkpoint and (base_model is not None):
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, device_map="auto", token_type_ids=None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", max_memory={0: "80GIB"}
            )
            self.model = PeftModel.from_pretrained(base_model_obj, model_name)
            self.model_name = base_model
            self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
            self.token_limit = 4096 if "Llama-2" in base_model else 2048
            return

        kwargs = {}
        eightbit = False

        if "llama" in model_name.lower():
            if "/" in model_name:
                org, model_part = model_name.split("/", 1)
                if model_part.endswith("-8bit"):
                    kwargs = {
                        "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
                    }
                    model_part = model_part[: -len("-8bit")]
                    eightbit = True
                else:
                    kwargs = {}
                    eightbit = False
                full_model_id = f"{org}/{model_part}"
            else:
                if model_name.endswith("-8bit"):
                    kwargs = {
                        "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
                    }
                    model_name = model_name[: -len("-8bit")]
                    eightbit = True
                else:
                    kwargs = {}
                    eightbit = False
                if "Llama-2" in model_name:
                    base = "meta-llama"
                    model_name = model_name + "-hf"
                elif "Llama-3" in model_name:
                    if "Bio-Medical" in model_name:
                        base = "ContactDoctor"
                    elif "medical" in model_name:
                        base = "Mlking2"
                    else:
                        base = "meta-llama"
                else:
                    base = "huggyllama"
                full_model_id = f"{base}/{model_name}"

            self.tokenizer = AutoTokenizer.from_pretrained(
                full_model_id, device_map="auto", token_type_ids=None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if is_peft_lora:
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    full_model_id, device_map="auto", max_memory={0: "80GIB"}, **kwargs
                )
                self.model = PeftModel.from_pretrained(
                    base_model_obj, full_model_id, is_trainable=True
                )
            else:
                if "/" in full_model_id and not is_peft_lora:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        full_model_id,
                        device_map="auto",
                        max_memory={0: "80GIB"},
                        **kwargs,
                    )
                else:
                    llama65b = "65b" in model_name and base == "huggyllama"
                    llama2_70b = "70b" in model_name and base == "meta-llama"
                    if ("7b" in model_name or "13b" in model_name) or eightbit:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            f"{base}/{model_name}",
                            device_map="auto",
                            max_memory={0: "80GIB"},
                            **kwargs,
                        )
                    elif llama2_70b or llama65b:
                        path = snapshot_download(
                            repo_id=f"{base}/{model_name}",
                            allow_patterns=["*.json", "*.model", "*.safetensors"],
                            ignore_patterns=["pytorch_model.bin.index.json"],
                        )
                        config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                        with accelerate.init_empty_weights():
                            self.model = AutoModelForCausalLM.from_config(config)
                        self.model.tie_weights()
                        max_mem = 15 * 4686198491
                        device_map = accelerate.infer_auto_device_map(
                            self.model.model,
                            max_memory={0: max_mem, 1: max_mem},
                            dtype="float16",
                        )
                        device_map = remove_split_layer(device_map)
                        full_model_device_map = {
                            f"model.{k}": v for k, v in device_map.items()
                        }
                        full_model_device_map["lm_head"] = 0
                        self.model = accelerate.load_checkpoint_and_dispatch(
                            self.model,
                            path,
                            device_map=full_model_device_map,
                            dtype="float16",
                            skip_keys="past_key_values",
                        )
                    else:
                        raise ValueError

        elif "mistral" in model_name.lower():
            if model_name.endswith("-8bit"):
                kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
                model_name = model_name[: -len("-8bit")]
            if model_name.endswith("-4bit"):
                kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
                model_name = model_name[: -len("-4bit")]
            else:
                kwargs = {}
            model_id = f"mistralai/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", max_memory={0: "80GIB"}, **kwargs
            )
        elif "falcon" in model_name:
            model_id = f"tiiuae/{model_name}"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                device_map="auto",
                token_type_ids=None,
                clean_up_tokenization_spaces=False,
            )
            kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device_map="auto", **kwargs
            )
        elif (
            "tinydolphin" in model_name.lower()
            or "tinyllama" in model_name.lower()
            or ("checkpoint" in model_name.lower() and os.path.isdir(model_name))
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto", token_type_ids=None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory={0: "80GIB"}
            )
            adapter_config_path = os.path.join(model_name, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                self.model = PeftModel.from_pretrained(base_model_obj, model_name)
            else:
                self.model = base_model_obj
        elif "ibm-granite" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto", token_type_ids=None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory={0: "80GIB"}
            )
        elif (
            "granite3-moe1b" in model_name.lower()
            or "granite3-moe" in model_name.lower()
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto", token_type_ids=None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory={0: "80GIB"}
            )

        elif "qwen" in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, device_map="auto", token_type_ids=None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory={0: "80GIB"}
            )
        else:
            raise ValueError(
                f"Unknown model branch for model_name `{model_name}` in HuggingfaceModel."
            )

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if "Llama-2" in model_name else 2048

    def predict(self, input_data, temperature, top_p, return_full=False):
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if (
            "llama" in self.model_name.lower()
            or "falcon" in self.model_name
            or "mistral" in self.model_name.lower()
        ):
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList(
                [
                    StoppingCriteriaSub(
                        stops=self.stop_sequences,
                        initial_length=len(inputs["input_ids"][0]),
                        tokenizer=self.tokenizer,
                    )
                ]
            )
        else:
            stopping_criteria = None

        logging.debug("temperature: %f", temperature)
        logging.debug("top_p: %f", top_p)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                "Generation exceeding token limit %d > %d",
                len(outputs.sequences[0]),
                self.token_limit,
            )

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        if return_full:
            return full_answer

        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            logging.warning(
                "Generated answer does not start with the input prompt; returning the full output as the answer."
            )
            input_data_offset = 0

        answer = full_answer[input_data_offset:]

        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                logging.warning(
                    "Stop words still present in the generated answer. Removing them manually."
                )
                for stop in self.stop_sequences:
                    sliced_answer = sliced_answer.replace(stop, "")

        sliced_answer = sliced_answer.strip()

        token_stop_index = self.tokenizer(
            full_answer[: input_data_offset + stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_input_token = len(inputs["input_ids"][0])
        n_generated = token_stop_index - n_input_token

        extracted_embeddings_dict = {}
        all_token_hidden_states = outputs.get("hidden_states")

        if all_token_hidden_states:
            num_generated_tokens_in_hs = len(all_token_hidden_states)

            if num_generated_tokens_in_hs == 0:
                logging.warning("Hidden states tuple from generate output is empty. Cannot extract embeddings.")
                for layer_spec_idx in self.probe_layers_to_extract:
                    extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = None
            else:
                last_token_hs_index = -1 

                last_token_all_layer_states = all_token_hidden_states[last_token_hs_index]

                if isinstance(last_token_all_layer_states, tuple):
                    num_layers_in_output_hs = len(last_token_all_layer_states)

                    for layer_spec_idx in self.probe_layers_to_extract:
                        actual_layer_idx_in_tuple = layer_spec_idx
                        if layer_spec_idx < 0:
                            actual_layer_idx_in_tuple = num_layers_in_output_hs + layer_spec_idx

                        if 0 <= actual_layer_idx_in_tuple < num_layers_in_output_hs:
                            layer_embedding_tensor = last_token_all_layer_states[actual_layer_idx_in_tuple]
                            if isinstance(layer_embedding_tensor, torch.Tensor):
                                if layer_embedding_tensor.ndim == 3 and layer_embedding_tensor.shape[1] == 1:
                                     layer_embedding_tensor = layer_embedding_tensor.squeeze(1)
                                extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = layer_embedding_tensor.cpu()
                            else:
                                logging.warning(f"Expected tensor for layer {layer_spec_idx} at last token, got {type(layer_embedding_tensor)}. Skipping.")
                                extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = None
                        else:
                            logging.warning(
                                f"Requested layer_spec_idx {layer_spec_idx} (maps to {actual_layer_idx_in_tuple}) is out of bounds "
                                f"for {num_layers_in_output_hs} available layer outputs for the last token. Skipping."
                            )
                            extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = None
                else:
                     logging.error(f"Expected a tuple of layer states for the last token, but got {type(last_token_all_layer_states)}. Format might differ. Cannot extract multi-layer embeddings.")
                     for layer_spec_idx in self.probe_layers_to_extract:
                         extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = None
        else:
            logging.warning("No 'hidden_states' found in model.generate() output. IS Probe embeddings will be None/missing.")
            for layer_spec_idx in self.probe_layers_to_extract:
                extracted_embeddings_dict[f"layer_{layer_spec_idx}"] = None

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1 and n_generated > 0:
            log_likelihoods = log_likelihoods
        elif n_generated > 0:
            log_likelihoods = log_likelihoods[:n_generated]
        else:
             log_likelihoods = []


        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation interrupted by max_token limit.")

        if len(log_likelihoods) == 0 and n_generated > 0 :
            logging.error("Log likelihoods are empty but n_generated > 0. This is unexpected.")


        return sliced_answer, log_likelihoods, extracted_embeddings_dict

    def get_p_true(self, input_data):
        input_data += " A"
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(
            "cuda"
        )["input_ids"]

        target_ids_true = tokenized_prompt_true.clone()
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(
                tokenized_prompt_true, labels=target_ids_true
            )

        loss_true = model_output_true.loss

        return -loss_true.item()


def get_parser(stages=["generate", "compute"]):
    entity = os.getenv("WANDB_SEM_UNC_ENTITY", None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep default wandb clean.",
    )
    parser.add_argument("--entity", type=str, default=entity)
    parser.add_argument("--random_seed", type=int, default=10)
    parser.add_argument(
        "--metric",
        type=str,
        default="squad",
        choices=["squad", "bertscore", "llm", "llm_gpt-3.5", "llm_gpt-4"],
        help="Metric to assign accuracy to generations.",
    )
    parser.add_argument(
        "--metric_threshold",
        type=float,
        default=0.5,
        help="Threshold for the chosen metric to determine correctness (e.g., 0.85 for bertscore).",
    )
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute accuracy at all temperatures or only t<<1.",
    )
    parser.add_argument(
        "--experiment_lot",
        type=str,
        default="Unnamed Experiment",
        help="Keep default wandb clean.",
    )
    if "generate" in stages:
        parser.add_argument(
            "--model_name",
            type=str,
            default="Llama-2-7b-chat",
            help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens",
            type=int,
            default=50,
            help="Max number of tokens generated.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="bioasq",
            help="Dataset to use",
        )
        parser.add_argument(
            "--ood_train_dataset",
            type=str,
            default=None,
            choices=["medquad", "bioasq"],
            help="Dataset to use to assemble few-shot prompt, p_true prompt, and train p_ik.",
        )
        parser.add_argument(
            "--num_samples", type=int, default=400, help="Number of samples to use"
        )
        parser.add_argument(
            "--num_few_shot",
            type=int,
            default=5,
            help="Number of few shot examples to use",
        )
        parser.add_argument(
            "--p_true_num_fewshot",
            type=int,
            default=20,
            help="Number of few shot examples to use",
        )
        parser.add_argument(
            "--p_true_hint",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?",
        )
        parser.add_argument(
            "--num_generations",
            type=int,
            default=10,
            help="Number of generations to use",
        )
        parser.add_argument(
            "--temperature", type=float, default=1.0, help="Temperature"
        )
        parser.add_argument("--top_p", type=float, default=0.9, help="Top p sampling")
        parser.add_argument(
            "--use_mc_options",
            type=bool,
            default=True,
            help="Include MC options question?",
        )
        parser.add_argument(
            "--get_training_set_generations",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?",
        )
        parser.add_argument(
            "--use_context",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?",
        )
        parser.add_argument(
            "--get_training_set_generations_most_likely_only",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Only get embedding of most likely answer for training set. "
                "This is all that's needed for p_true."
            ),
        )
        parser.add_argument(
            "--compute_p_true", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument("--brief_prompt", default="default", type=str)
        parser.add_argument("--prompt_type", default="default", type=str)
        parser.add_argument(
            "--compute_uncertainties",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Trigger compute_uncertainty_measures.py",
        )
        parser.add_argument(
            "--answerable_only",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Exclude unanswerable questions.",
        )

        parser.add_argument(
            "--base_model",
            type=str,
            default=None,
            help="Base model name to use for adapter checkpoints (e.g., 'TinyLlama/TinyLlama_v1.1').",
        )

        parser.add_argument(
            "--summarize_medquad",
            action="store_true",
            help="Whether to summarize long MedQuad questions before loading.",
        )
        parser.add_argument(
            "--probe_layers_to_extract",
            type=int,
            nargs="+",
            default=[-1],
            help="List of layer indices (negative for from_end) to extract embeddings for IS probe.",
        )


    if "compute" in stages:
        parser.add_argument(
            "--recompute_accuracy", default=False, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--eval_wandb_runid",
            type=str,
            help="wandb run id of the dataset to evaluate on",
        )
        parser.add_argument(
            "--train_wandb_runid",
            type=str,
            default=None,
            help="wandb run id of the dataset from which training embeddings and p_true samples will be taken",
        )
        parser.add_argument("--num_eval_samples", type=int, default=int(1e19))
        parser.add_argument(
            "--compute_predictive_entropy",
            default=True,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--compute_p_ik", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--compute_p_ik_answerable",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--compute_context_entails_response",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--analyze_run", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--assign_new_wandb_id", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument("--restore_entity_eval", type=str, default=entity)
        parser.add_argument("--restore_entity_train", type=str, default=entity)
        parser.add_argument(
            "--condition_on_question",
            default=True,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--strict_entailment", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument(
            "--use_all_generations", default=True, action=argparse.BooleanOptionalAction
        )
        parser.add_argument("--use_num_generations", type=int, default=-1)
        parser.add_argument("--entailment_model", default="deberta", type=str)
        parser.add_argument(
            "--entailment_cache_id",
            default=None,
            type=str,
            help="Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.",
        )
        parser.add_argument(
            "--entailment_cache_only",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--compute_p_true_in_compute_stage",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser.add_argument(
            "--reuse_entailment_model",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Use entailment model as p_true model.",
        )
    return parser

def setup_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(logging.INFO)

def construct_fewshot_prompt_from_indices(
    dataset, example_indices, brief, brief_always, make_prompt
):
    if not brief_always:
        prompt = brief
    else:
        prompt = ""

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt

def split_dataset(dataset):

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    assert set(answerable_indices) | set(unanswerable_indices) == set(
        range(len(dataset))
    )
    assert set(answerable_indices) - set(unanswerable_indices) == set(
        answerable_indices
    )

    return answerable_indices, unanswerable_indices

def model_based_metric(predicted_answer, example, model):
    if "answers" in example:
        correct_answers = example["answers"]["text"]
    elif "reference" in example:
        correct_answers = example["reference"]["answers"]["text"]
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += (
            f"The following are expected answers to this question: {correct_answers}.\n"
        )

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if "gpt" in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01, top_p=1.0)

    if "yes" in predicted_answer.lower():
        return 1.0
    elif "no" in predicted_answer.lower():
        return 0.0
    else:
        logging.warning("Redo llm check.")
        predicted_answer, _, _ = model.predict(prompt, 1, top_p=1.0)
        if "yes" in predicted_answer.lower():
            return 1.0
        elif "no" in predicted_answer.lower():
            return 0.0

        logging.warning("Answer neither no nor yes. Defaulting to no!")
        return 0.0

def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)

def get_gpt_metric(metric_name):

    model_name = "_".join(metric_name.split("_")[1:])

    class EntailmentGPT:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            # Assume oai.predict exists and works
            pass
            # return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric

def get_reference(example):
    if "answers" not in example and "reference" in example:
        example = example["reference"]
    answers = example.get("answers", {})
    answer_starts = answers.get("answer_start", [])
    text = answers.get("text", [])
    reference = {
        "answers": {"answer_start": answer_starts, "text": text},
        "id": example.get("id", None),
    }
    return reference

_metric_cache = {}

def get_metric(metric_name, device="cuda"):
    global _metric_cache

    if metric_name in _metric_cache:
        return _metric_cache[metric_name]

    logging.info(f"Loading metric: {metric_name}")
    metric_fn = None

    if metric_name == "squad":
        try:
            squad_metric = load("squad_v2")
            def squad_metric_fn(response, example, *args, **kwargs):
                if "id" in example: exid = example["id"]
                elif "reference" in example and "id" in example["reference"]: exid = example["reference"]["id"]
                else: raise ValueError("Missing ID for SQuAD metric")

                prediction = {"prediction_text": response, "no_answer_probability": 0.0, "id": exid}
                reference = get_reference(example)
                results = squad_metric.compute(predictions=[prediction], references=[reference])
                return results.get("f1", 0.0) / 100.0
            metric_fn = squad_metric_fn
        except Exception as e:
            logging.error(f"Failed to load SQuAD metric: {e}")
            raise

    elif metric_name == "bertscore":
        try:
            bertscore_metric = load("bertscore")
            def bertscore_metric_fn(response, example, *args, **kwargs):
                reference = get_reference(example)
                references_list = reference['answers']['text']
                if not references_list:
                    return 0.0
                try:
                    predictions_list = [response] * len(references_list)
                    results = bertscore_metric.compute(
                        predictions=predictions_list,
                        references=references_list,
                        lang="en",
                        model_type="microsoft/deberta-v2-xlarge-mnli",
                        device=device
                    )
                    f1_scores = results.get("f1")
                    if f1_scores and isinstance(f1_scores, list) and len(f1_scores) > 0:
                         max_f1 = float(np.max(f1_scores))
                    else:
                         max_f1 = 0.0
                    return max_f1
                except Exception as e_bs:
                    logging.error(f"BERTScore computation failed for ID {example.get('id', 'N/A')}: {e_bs}")
                    return 0.0
            metric_fn = bertscore_metric_fn
        except Exception as e:
            logging.error(f"Failed to load BERTScore metric: {e}")
            raise

    elif "llm_gpt" in metric_name:
        metric_fn = get_gpt_metric(metric_name)
    elif metric_name == "llm":
        metric_fn = llm_metric
    else:
        raise ValueError(f"Unsupported metric specified: {metric_name}")

    if metric_fn:
        _metric_cache[metric_name] = metric_fn
        logging.info(f"Metric '{metric_name}' loaded successfully.")
    return metric_fn


def init_model(args):
    mn = args.model_name
    if (
        "llama" in mn.lower()
        or "falcon" in mn.lower()
        or "mistral" in mn.lower()
        or "tinydolphin" in mn.lower()
        or "granite" in mn.lower()
        or "qwen" in mn.lower()
        or ("checkpoint" in mn.lower() and os.path.isdir(mn))
    ):
        model_kwargs = {
             "stop_sequences": "default",
             "max_new_tokens": args.model_max_new_tokens,
             "probe_layers_to_extract": getattr(args, 'probe_layers_to_extract', [-1])
        }
        if args.base_model is not None:
            model_kwargs["base_model"] = args.base_model

        model = HuggingfaceModel(mn, **model_kwargs)
    else:
        raise ValueError(f"Unknown model_name `{mn}`.")
    return model

def get_make_prompt(args):
    if args.prompt_type == "default":

        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ""
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += "Answer:"
            return prompt

    else:
        raise ValueError

    return make_prompt


def save(object_to_save, file):
    with open(f"{wandb.run.dir}/{file}", "wb") as f:
        pickle.dump(object_to_save, f)
    wandb.save(f"{wandb.run.dir}/{file}")

def load_pickle(filepath):
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
        logging.error(f"Error loading pickle file {filepath}: {e}", exc_info=True)
        return None

def save_pickle(data_object, filepath):
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data_object, f)
        logging.info(f"Successfully saved object to {filepath}")
    except Exception as e:
        logging.error(f"Error saving pickle file {filepath}: {e}", exc_info=True)