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
    def predict(self, input_data, temperature, top_p, return_full=False):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, tokenizer, match_on="text", initial_length=None):
        """
        Initialize StoppingCriteriaSub.

        Args:
            stops (List[str]): List of strings which, when encountered, will stop generation.
            tokenizer (transformers.AutoTokenizer): Tokenizer to use for encoding.
            match_on (str, optional): Whether to match on "text" or "tokens". Defaults to "text".
            initial_length (int, optional): Initial sequence length. Defaults to None.
        """
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
        """
        Check if the generated sequence matches any of the stopping criteria.

        This method examines the generated sequence of tokens or text to determine if it contains
        any predefined stopping sequences. If a match is found, it returns True, indicating that
        the generation should stop; otherwise, it returns False.

        Args:
            input_ids (torch.LongTensor): The token IDs of the generated sequence.
            scores (torch.FloatTensor): The scores of the generated tokens (unused).

        Returns:
            bool: True if any stopping sequence is matched, False otherwise.

        Raises:
            ValueError: If an invalid match_on criterion is specified.
        """

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
    """
    Given a device map, remove split layers (layers with two or more keys)
    and replace them with a single key. This is useful for models that have
    split layers, which can be challenging to work with.

    Args:
        device_map_in: A dictionary mapping layer names to devices.

    Returns:
        A new device map with split layers removed.
    """
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
        self, model_name, stop_sequences=None, max_new_tokens=None, base_model=None, probe_layers_to_extract=None
    ):
        """
        Initialize HuggingfaceModel.

        Args:
            model_name (str): Model name from Hugging Face Hub.
            stop_sequences (List[str], optional): List of strings that will stop generation. Defaults to STOP_SEQUENCES.
            max_new_tokens (int): Maximum number of new tokens to generate.
            base_model (str, optional): Base model name for PEFT adapter. Defaults to None.
            probe_layers_to_extract (List[int], optional): List of layer indices to extract and probe. Defaults to [-1, -2, -4].
        """
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be provided.")
        self.max_new_tokens = max_new_tokens

        if probe_layers_to_extract is None:
            self.probe_layers_to_extract = [-1, -2, -4] 
            logging.info(f"No probe_layers_to_extract specified, defaulting to: {self.probe_layers_to_extract}")
        else:
            self.probe_layers_to_extract = probe_layers_to_extract
            logging.info(f"Using specified probe_layers_to_extract: {self.probe_layers_to_extract}")

        if stop_sequences == "default":
            stop_sequences = STOP_SEQUENCES

        is_peft_lora = False
        try:
            _ = PeftConfig.from_pretrained(model_name)
            is_adapter_checkpoint = True
        except Exception as e:
            is_adapter_checkpoint = False
        
        self.model_name = model_name
        
        self.tokenizer = None
        self.model = None

        actual_model_id_to_load = model_name
        is_peft_model = False
        if is_adapter_checkpoint:
            if base_model:
                actual_model_id_to_load = base_model
                is_peft_model = True
                logging.info(f"PEFT adapter {model_name} will be loaded onto base model {base_model}.")
            else:
                try:
                    peft_config = PeftConfig.from_pretrained(model_name)
                    actual_model_id_to_load = peft_config.base_model_name_or_path
                    is_peft_model = True
                    logging.info(f"Inferred base model {actual_model_id_to_load} from adapter config for {model_name}")
                except Exception as e:
                    logging.error(f"Could not infer base model for adapter {model_name}: {e}. Please provide --base_model argument or ensure adapter_config.json is correct.")
                    raise ValueError(f"Base model required for PEFT adapter {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_id_to_load, device_map="auto", token_type_ids=None)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_kwargs = {}
        temp_model_name_for_quant_check = model_name
        if is_peft_model:
             temp_model_name_for_quant_check = actual_model_id_to_load

        if temp_model_name_for_quant_check.endswith("-8bit"):
            quantization_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
            logging.info(f"Applying 8-bit quantization for {actual_model_id_to_load}")
        elif temp_model_name_for_quant_check.endswith("-4bit"):
             quantization_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
             logging.info(f"Applying 4-bit quantization for {actual_model_id_to_load}")
        
        device_map_config = "auto"
        max_memory_config = {0: "80GIB"}
        is_large_llama = ("llama-2-70b" in actual_model_id_to_load.lower() or "llama-65b" in actual_model_id_to_load.lower()) and not quantization_kwargs
        
        if is_large_llama:
             logging.info(f"Attempting special loading for large Llama model: {actual_model_id_to_load}")
             path = snapshot_download(
                repo_id=actual_model_id_to_load,
                allow_patterns=["*.json", "*.model", "*.safetensors"],
                ignore_patterns=["pytorch_model.bin.index.json"],
            )
             config = AutoConfig.from_pretrained(actual_model_id_to_load)
             with accelerate.init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(config)
             self.model.tie_weights()
             max_mem_gb = 70
             max_mem_bytes = max_mem_gb * 1024**3 
             device_map = accelerate.infer_auto_device_map(
                self.model.model,
                max_memory={i: max_mem_bytes for i in range(torch.cuda.device_count())} if torch.cuda.device_count() > 0 else {0:max_mem_bytes},
                dtype="float16", 
                no_split_module_classes=self.model._no_split_modules
            )
             full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
             if hasattr(self.model, 'lm_head'):
                full_model_device_map["lm_head"] = device_map.get(list(device_map.keys())[0], 0)
             
             self.model = accelerate.load_checkpoint_and_dispatch(
                self.model,
                path,
                device_map=full_model_device_map,
                dtype="float16",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                actual_model_id_to_load,
                device_map=device_map_config,
                max_memory=max_memory_config,
                trust_remote_code=True,
                **quantization_kwargs
            )

        if is_peft_model:
            self.model = PeftModel.from_pretrained(self.model, model_name)
            logging.info(f"Loaded PEFT adapter {model_name} onto {actual_model_id_to_load}.")

        self.model_name = actual_model_id_to_load
        self.stop_sequences = (stop_sequences or []) + [self.tokenizer.eos_token]
        
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.token_limit = self.model.config.max_position_embeddings
        elif "llama" in self.model_name.lower():
            self.token_limit = 4096 if "llama-2" in self.model_name.lower() or "llama-3" in self.model_name.lower() else 2048
        else:
            self.token_limit = 2048 
        logging.info(f"Model token limit set to: {self.token_limit}")
        
        self.hidden_size = None 
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_size = self.model.config.hidden_size
            logging.info(f"Determined model hidden_size: {self.hidden_size}")
        else:
            logging.warning("Could not determine model's hidden_size from config. Zero-padding for missing IS probe layers might fail.")


    def predict(self, input_data, temperature, top_p, return_full=False, do_sample=True):
        """
        Predict with the model.

        Args:
            input_data: Input text or prompt
            temperature: Temperature for sampling
            top_p: Top-p value for sampling
            return_full: Whether to return the full generated text or just the portion after the input
            do_sample: Whether to sample or take the argmax at each step

        Returns:
            The generated text, log likelihoods of the generated tokens, and the last token's embedding
        """
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        input_token_len = inputs['input_ids'].shape[1]

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=input_token_len,
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        logging.debug('top_p: %f', top_p)
        logging.debug("do_sample: %s", do_sample)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if outputs.sequences[0].shape[0] > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            return full_answer

        generated_token_ids = outputs.sequences[0][input_token_len:]

        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            earliest_stop_index = len(answer)
            for stop in self.stop_sequences:
                stop_index = sliced_answer.find(stop)
                if stop_index != -1:
                    earliest_stop_index = min(earliest_stop_index, stop_index)

            if earliest_stop_index < len(answer):
                sliced_answer = sliced_answer[:earliest_stop_index]

            for stop in self.stop_sequences:
                 if sliced_answer.endswith(stop):
                      sliced_answer = sliced_answer[:-len(stop)]


        sliced_answer = sliced_answer.strip()

        final_generated_token_ids = self.tokenizer.encode(sliced_answer, add_special_tokens=False)
        n_generated = len(final_generated_token_ids)

        if n_generated <= 0:
            logging.warning('Only stop_words or empty string generated after prompt removal. Using first generated token for likelihoods/embeddings.')
            n_generated = 1
            if len(outputs.scores) == 0:
                 logging.error("Model generated absolutely nothing, cannot compute likelihoods/embeddings.")
                 return sliced_answer, [], None

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        log_likelihoods = transition_scores[0][:n_generated].tolist()

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        elif 'hidden_states' in outputs.keys():
            hidden = outputs.hidden_states
        else:
            logging.warning("No hidden states found in model output, cannot extract embedding.")
            hidden = None
            last_token_embedding = None

        if hidden is not None:
            hidden_state_index = n_generated - 1

            if hidden_state_index < 0:
                 logging.warning("n_generated <= 0, using first hidden state for embedding.")
                 hidden_state_index = 0

            if hidden_state_index >= len(hidden):
                logging.warning(f"Required hidden state index {hidden_state_index} is out of bounds (len={len(hidden)}). Using last available hidden state.")
                hidden_state_index = len(hidden) - 1

            if hidden_state_index < 0:
                 logging.error("Cannot get embedding: No hidden states available in output.")
                 last_token_embedding = None
            else:
                 last_input = hidden[hidden_state_index]
                 last_layer = last_input[-1]
                 last_token_embedding = last_layer[:, -1, :].cpu()

        if len(log_likelihoods) == 0 and n_generated > 0:
             logging.error(f"Log likelihoods list is empty despite n_generated={n_generated}. Check transition score calculation.")
             log_likelihoods = []

        if 'last_token_embedding' not in locals():
             last_token_embedding = None

        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """
        Computes the log probability of the input data (prompt) being true, by
        prepending ' A' to the input and computing the log likelihood of the
        resulting sequence.

        Args:
            input_data (str): The input data (prompt) to compute the log probability of.

        Returns:
            float: The log probability of the input data being true.
        """
        input_data += " A"
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors="pt").to(self.model.device)

        target_ids_true = tokenized_prompt_true["input_ids"].clone()
        target_ids_true[:, :-1] = -100 

        with torch.no_grad():
            model_output_true = self.model(
                **tokenized_prompt_true, labels=target_ids_true
            )
        loss_true = model_output_true.loss
        return -loss_true.item()