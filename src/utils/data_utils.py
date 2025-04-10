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

# pylint: disable=missing-function-docstring


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
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
        else:
            kagglehub.login()

        path = kagglehub.dataset_download("jpmiller/layoutlm")
        csv_path = os.path.join(path, "medquad.csv")
        df = pd.read_csv(csv_path)

        do_summarize_answers = (
            add_options.get("summarize_answers", False) if add_options else False
        )

        if do_summarize_answers:
            summarizer_model = add_options.get("summarizer_model", "t5-small")
            print(
                f"[MedQuad loader] Summarizing ANSWERS with model: {summarizer_model}"
            )

            summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model)

            def summarize_text(text, max_len=150):
                """Use T5 to summarize text to ~max_len tokens (approx)."""
                orig_tokens = len(text.split())

                inputs = summarizer_tokenizer(
                    "summarize: " + text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).input_ids
                outputs = summarizer_model.generate(
                    inputs, max_length=max_len, num_beams=2, no_repeat_ngram_size=2
                )
                summary = summarizer_tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                summary_tokens = len(summary.split())

                if orig_tokens > 40:
                    print("[DEBUG] Summarizing answer:")
                    print(f"  Original (first 150 chars) => {text[:150]}...")
                    print(f"  Original token count:      {orig_tokens}")
                    print(f"  Summary (first 150 chars) => {summary[:150]}...")
                    print(f"  Summary token count:       {summary_tokens}")
                    print("----------")

                return summary

        else:

            def summarize_text(text, max_len=150):
                return text

        if add_options is not None and "num_samples" in add_options:
            df = df.sample(n=add_options["num_samples"], random_state=seed)

        dataset_dict = {"question": [], "answers": [], "id": [], "context": []}
        dropped = 0

        for _, row in df.iterrows():
            ans = row.get("exact_answer", None)
            if pd.isna(ans):
                ans = row.get("answer", None)
            if pd.isna(ans):
                dropped += 1
                continue

            question = clean_text(row.get("question", ""), max_len=max_question_len)

            all_answers = [a.strip() for a in ans.split(";") if a.strip()]

            summarized_answers = []
            for single_ans in all_answers:
                single_ans = clean_text(single_ans, max_len=2000)
                if do_summarize_answers and len(single_ans.split()) > 30:
                    single_ans = summarize_text(single_ans, max_len=100)
                single_ans = clean_text(single_ans, max_len=300)
                summarized_answers.append(single_ans)

            if not summarized_answers:
                dropped += 1
                continue

            dataset_dict["question"].append(question)
            dataset_dict["answers"].append(
                {
                    "text": summarized_answers,
                    "answer_start": [0] * len(summarized_answers),
                }
            )

            if "id" in row and pd.notna(row["id"]):
                dataset_dict["id"].append(str(row["id"]))
            else:
                unique_id = hashlib.md5(
                    (question + " ".join(summarized_answers)).encode("utf-8")
                ).hexdigest()
                dataset_dict["id"].append(unique_id)

            context_str = row.get("context", "")
            context_str = clean_text(context_str, max_len=max_context_len)
            dataset_dict["context"].append(context_str)

        if dropped > 0:
            print(f"Dropped {dropped} rows in MedQuad due to missing or empty answers.")

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

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
