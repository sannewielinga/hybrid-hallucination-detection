"""Implement semantic entropy."""

import os
import pickle
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils.models import HuggingfaceModel
from src.utils import utils


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):

    def __init__(
        self,
        model_name="lighteternal/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-finetuned-mnli",
    ):
        logging.info(f"Loading Entailment Model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            DEVICE
        )
        self.entailment_idx = 0
        self.neutral_idx = 1
        self.contradiction_idx = 2

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True, max_length=512
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction_idx = torch.argmax(probs).cpu().item()

        if prediction_idx == self.entailment_idx:
            return 2
        elif prediction_idx == self.neutral_idx:
            return 1
        elif prediction_idx == self.contradiction_idx:
            return 0
        else:
            logging.warning(
                f"Unexpected prediction index {prediction_idx} from NLI model."
            )
            return 1


class EntailmentLLM(BaseEntailment):

    entailment_file = "entailment_cache.pkl"

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info("Restoring prediction cache from %s", entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir
        )

        with open(f"{wandb.run.dir}/{self.entailment_file}", "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        utils.save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example["question"])

        logging.info("%s input: %s", self.name, prompt)

        hashed = oai.md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info("Restoring hashed instead of predicting with model.")
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info("%s prediction: %s", self.name, response)

        binary_response = response.lower()[:30]
        if "entailment" in binary_response:
            return 2
        elif "neutral" in binary_response:
            return 1
        elif "contradiction" in binary_response:
            return 0
        else:
            logging.warning("MANUAL NEUTRAL!")
            return 1


class EntailmentGPT4(EntailmentLLM):

    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = "gpt-4"

    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += (
            "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."
            ""
        )

        return prompt

    def predict(self, prompt, temperature):
        return oai.predict(prompt, temperature, model=self.name)


class EntailmentGPT35(EntailmentGPT4):

    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = "gpt-3.5"


class EntailmentGPT4Turbo(EntailmentGPT4):

    def __init__(self, entailment_cache_id, entailment_cache_only):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = "gpt-4-turbo"


class EntailmentLlama(EntailmentLLM):

    def __init__(self, entailment_cache_id, entailment_cache_only, name):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = name
        self.model = HuggingfaceModel(name, stop_sequences="default", max_new_tokens=30)

    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += (
            "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"
            ""
        )
        prompt += "Response:" ""

        return prompt

    def predict(self, prompt, temperature):
        predicted_answer, _, _ = self.model.predict(prompt, temperature)
        return predicted_answer


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(
            text2, text1, example=example
        )  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            semantically_equivalent = (0 not in implications) and (
                [1, 1] != implications
            )

        return semantically_equivalent

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized"):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == "sum_normalized":
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = -(probabilities * np.log(probabilities)).sum()
    return entropy
