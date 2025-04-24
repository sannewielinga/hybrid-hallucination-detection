"""Implement HuggingfaceModel models."""

import copy
import logging
from collections import Counter
import torch
import os

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download

from peft import PeftModel
from peft import PeftConfig

from abc import ABC, abstractmethod
from typing import List, Text


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
    """Stop generations when they match a particular text or token."""

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
    """Modify device maps s.t. individual layers are not spread across devices."""

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
    """Hugging Face Model."""

    def __init__(
        self, model_name, stop_sequences=None, max_new_tokens=None, base_model=None
    ):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be provided.")
        self.max_new_tokens = max_new_tokens

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

        # --- Llama branch ---
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
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", max_memory={0: "80GIB"}
            )
            adapter_config_path = os.path.join(model_name, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                self.model = PeftModel.from_pretrained(base_model, model_name)
            else:
                self.model = base_model
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

        if n_generated == 0:
            logging.warning(
                "Only stop_words were generated. For likelihoods and embeddings, taking stop word instead."
            )
            n_generated = 1

        if "decoder_hidden_states" in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                "Taking first and only generation for hidden! "
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
            )
            last_input = hidden[0]
        elif (n_generated - 1) >= len(hidden):
            logging.error(
                "Taking last state because n_generated is too large"
                "n_generated: %d, n_input_token: %d, token_stop_index %d, "
                "last_token: %s, generation was: %s, slice_answer: %s",
                n_generated,
                n_input_token,
                token_stop_index,
                self.tokenizer.decode(outputs["sequences"][0][-1]),
                full_answer,
                sliced_answer,
            )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        last_layer = last_input[-1]
        last_token_embedding = last_layer[:, -1, :].cpu()

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning("Taking first and only generation for log likelihood!")
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning("Generation interrupted by max_token limit.")

        if len(log_likelihoods) == 0:
            raise ValueError

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""

        input_data += " A"
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(
            "cuda"
        )["input_ids"]
        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.

        target_ids_true = tokenized_prompt_true.clone()
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(
                tokenized_prompt_true, labels=target_ids_true
            )

        loss_true = model_output_true.loss

        return -loss_true.item()
