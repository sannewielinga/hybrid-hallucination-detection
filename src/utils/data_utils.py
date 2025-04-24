"""Data loading utilities for BioASQ, MedQuAD, and MedQA."""

import os
import json
import hashlib
import datasets
import kagglehub
import pandas as pd
import re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


load_dotenv("./.env")


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset for medical QA: BioASQ, MedQuAD, and MedQA.

    Args:
        dataset_name (str): Name of the dataset ("bioasq", "medquad", or "medqa").
        seed (int): Random seed for reproducibility during train/test split.
        add_options: Dictionary with options like num_samples, max_question_len,
                     and max_context_len.

    Returns:
        tuple: (train_dataset, validation_dataset) as Huggingface Datasets.
    """
    train_dataset, validation_dataset = None, None

    max_question_len = add_options.get("max_question_len", 300) if add_options else 300
    max_context_len = add_options.get("max_context_len", 800) if add_options else 800

    def clean_text(text, max_len=None):
        """Clean text by removing extra whitespace and truncating to max_len."""
        if not isinstance(text, str):
            return ""
        text = " ".join(text.split())
        if max_len is not None and len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        return text

    if dataset_name.lower() == "bioasq":
        path = "data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {"question": [], "answers": [], "id": [], "context": []}

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:
                if isinstance(question["exact_answer"], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question["exact_answer"]
                    ]
                else:
                    exact_answers = [question["exact_answer"]]
                dataset_dict["answers"].append(
                    {"text": exact_answers, "answer_start": [0] * len(exact_answers)}
                )
            else:
                dataset_dict["answers"].append(
                    {"text": question["ideal_answer"], "answer_start": [0]}
                )
            dataset_dict["id"].append(question["id"])
            dataset_dict["context"].append(None)

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        print("Preview of processed bioasq training samples:")
        for i, sample in enumerate(train_dataset.select(range(5))):
            print(f"Sample {i}: {sample}")

    elif dataset_name.lower() == "medquad":
        path = "data/medquad/medquad_processed_concise.csv"
        logging.info(f"Attempting to load MedQuAD data from: {path}")
        try:
            data = pd.read_csv(path)
            logging.info(f"Loaded MedQuAD CSV with columns: {data.columns.tolist()}")
        except Exception as e:
            logging.error(f"Failed to load MedQuAD CSV file {path}: {e}")
            return None, None

        required_cols = ["Question", "ConciseAnswer_Gemini"]
        if not all(col in data.columns for col in required_cols):
             logging.error(f"MedQuAD CSV missing required columns. Found: {data.columns.tolist()}, Required: {required_cols}")
             return None, None

        dataset_dict = {"question": [], "answers": [], "id": [], "context": []}

        for index, row in data.iterrows():
            question_text = row["Question"]
            concise_answer_text = row["ConciseAnswer_Gemini"]

            cleaned_question = clean_text(question_text, max_len=max_question_len)
            answer_texts = []
            if pd.notna(concise_answer_text) and isinstance(concise_answer_text, str) and concise_answer_text.strip():
                cleaned_answer = clean_text(concise_answer_text, max_len=max_context_len)
                answer_texts.append(cleaned_answer)

            entry_id = f"medquad_{index}"

            dataset_dict["question"].append(cleaned_question)
            dataset_dict["answers"].append(
                 {"text": answer_texts, "answer_start": [0] * len(answer_texts)}
            )
            dataset_dict["id"].append(entry_id)
            dataset_dict["context"].append(None)

        if not dataset_dict["question"]:
             logging.error("No valid questions processed from MedQuAD file.")
             return None, None

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        logging.info(f"Successfully processed {len(train_dataset)} training and {len(validation_dataset)} validation samples from MedQuAD.")
        print("Preview of processed medquad training samples:")
        for i, sample in enumerate(train_dataset.select(range(min(5, len(train_dataset))))):
             print(f"Sample {i}: {sample}")

    elif dataset_name.lower() == "rag-mini-bioasq":
        ds = datasets.load_dataset(
            "enelpol/rag-mini-bioasq", "question-answer-passages"
        )
        dataset_dict = {"question": [], "answers": [], "id": [], "context": []}
        for ex in ds["train"]:
            question = clean_text(ex.get("question", ""), max_len=max_question_len)
            answer = ex.get("answer", "").strip()
            dataset_dict["question"].append(question)
            dataset_dict["answers"].append({"text": [answer], "answer_start": [0]})
            dataset_dict["id"].append(
                str(ex.get("id", f"ragbioasq_{len(dataset_dict['question'])}"))
            )
            dataset_dict["context"].append(None)
        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

    else:
        raise ValueError("Dataset not supported.")

    print(
        f"Loaded {len(train_dataset)} training examples and {len(validation_dataset)} validation examples for {dataset_name}."
    )
    return train_dataset, validation_dataset
