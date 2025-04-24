import google.generativeai as genai
import argparse
import json
import logging
import os
import time
import pandas as pd
from pathlib import Path
import re
import numpy as np


PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in analyzing and classifying errors (hallucinations) made by other language models in the medical domain. Your goal is to provide the **MOST specific and accurate classification** based on the provided taxonomy and a rigorous step-by-step process. **Strongly prefer specific A/B subtypes over '_Other' categories.**

**Taxonomy:**

**1. Factuality Errors:** The generated content contradicts established, verifiable real-world medical knowledge or facts *external* to the information implicitly or explicitly provided in the `question`.
    *   **A1_DosageAdministration:** Incorrect medical dosage, frequency, route, procedure.
    *   **A2_ContraindicationIndication:** Wrongly states safe/unsafe or indicated/contraindicated for the subject of the question.
    *   **A3_DiagnosticCriteriaDefinition:** Misrepresents established medical criteria, definitions, symptoms, timelines, or the established function/purpose of a medical concept/method **relevant to the question**.
    *   **B1_StatisticalEpidemiological:** Fabricates/misstates medical stats, efficacy, risk factors related to the question's topic.
    *   **B2_FabricatedEntityGuideline:** Invents non-existent medical entities, guidelines, studies related to the question's topic.

**2. Faithfulness Errors:** The generated content is inconsistent with the implicit or explicit information, constraints, or topic defined **within the `question` itself**. It ignores, contradicts, or misuses elements of the query.
    *   **A1_ContextIgnorant (Question-Based):** Overlooks or contradicts key entities, relationships, or constraints *mentioned or implied within the question*. (e.g., answering about the wrong drug when the question specified one).
    *   **A2_InstructionMisinterpretation:** Fails to follow the specific *task* or *format* instructed by the question (e.g., providing an explanation when asked to list, giving a single item when asked for multiple).
    *   **B1_ExtrapolationAddition (Question-Based):** Adds significant details or makes claims about the subject of the question that are not directly addressed or reasonably inferable *from the scope defined by the question*, even if factually plausible in a broader sense.

**3. Other/Unclear:** The error type is ambiguous, a mix of types, or doesn't fit the above categories clearly.

**--- Analysis Task ---**

**Instructions:** Follow these steps precisely:
1.  Analyze the 'Incorrect Generated Answer' considering the 'Question' and the 'Correct/Reference Answer(s)'. Treat the **question itself as the primary source of context and constraints**.
2.  **Step 1: Determine Primary Error Type.**
    *   First, evaluate **Faithfulness**: Does the answer directly address the entities, constraints, and task defined *by the question*? Does it ignore parts of the question (A1)? Does it fail the *task* requested (A2)? Does it go beyond the question's scope (B1)?
    *   If the answer *is* faithful to the question's topic and constraints but contains incorrect information based on external knowledge, evaluate **Factuality**.
    *   If neither fits well, consider **Other/Unclear**.
3.  **Step 2: Consider Specific Subtypes.**
    *   If **Factuality**: Methodically check A1-B2 based on external knowledge related to the question's subject. **Critically evaluate if one of A1-B2 applies.**
    *   If **Faithfulness**: Methodically check A1 (ignoring question elements), A2 (wrong task/format), or B1 (adding irrelevant details beyond question scope). **Critically evaluate if one of A1-B1 applies.**
4.  **Step 3: Select Final Code.**
    *   Choose the **single most specific A/B code** identified in Step 2 that accurately describes the core error mechanism.
    *   If the primary type was Other/Unclear, select that.
5.  **Step 4: Provide Rationale.**
    *   Justify your chosen code by explaining how the error violates external facts (Factuality) OR fails to adhere to the question's implicit/explicit constraints (Faithfulness).
    *   Keep the rationale concise and brief.

**Case Details:**

*   **Question:** {question}
*   **Correct/Reference Answer(s):** {reference_answers}
*   **Incorrect Generated Answer:** {generated_answer}

**--- Classification Output ---**

**Selected Subtype Code:** [Provide the SINGLE specific A/B code OR Other_Unclear. **Strongly prefer specific A/B codes.**]
**Rationale:** [Follow instructions in Step 4 above.]
"""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_NON_RATE_LIMIT_RETRIES = 5 
MAX_SLEEP_INTERVAL = 300

def format_prompt(template, details):
    """
    Formats the prompt template with the provided details.

    Modifies the details dictionary in-place by:
        - Joining the reference answers if they are a list
        - Replacing None reference answers with "N/A"
        - Replacing None generated answers with "[GENERATION FAILED]"

    Args:
        template (str): The prompt template with placeholders for the details.
        details (dict): The dictionary of details to be formatted into the prompt template.

    Returns:
        str: The formatted prompt string. If a KeyError occurs (i.e. a placeholder is missing), logs an error and returns None.
    """
    if isinstance(details.get('reference_answers'), list):
        details['reference_answers'] = "\n".join(details['reference_answers'])
    elif details.get('reference_answers') is None:
         details['reference_answers'] = "N/A"
    if details.get('generated_answer') is None:
         details['generated_answer'] = "[GENERATION FAILED]"
    try:
        return template.format(**details)
    except KeyError as e:
        logging.error(f"Missing placeholder in template: {e} for details: {details}")
        return None

def parse_llm_response(response_text):
    """
    Parse the LLM response text for the selected error code and rationale.

    Matches the code and rationale using regex patterns. If the code does not match
    the expected pattern, logs a warning. If either the code or rationale is not found,
    logs a warning and returns None for both.

    Returns:
        tuple: (code, rationale)
    """
    code = None
    rationale = None
    code_match = re.search(r"\*\*?Selected Subtype Code:\*\*?\s*(\S+)", response_text, re.IGNORECASE)
    rationale_match = re.search(r"\*\*?Rationale:\*\*?\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)

    if code_match:
        code = code_match.group(1).strip().replace('**', '')
    if rationale_match:
        rationale = rationale_match.group(1).strip().replace('**', '')
    if code and not re.match(r"^[AB][1-3]_[A-Za-z]+(\s*\(.*\))?|^(Factuality|Faithfulness)_Other|Other/Unclear$", code):
         logging.warning(f"Parsed code '{code}' seems invalid according to basic pattern check.")

    if code is None or rationale is None:
        logging.warning(f"Parsing failed for code or rationale in response: {response_text[:150]}...")
        return None, None

    return code, rationale

def automate_annotations(input_jsonl, sampled_ids_file, output_csv, model_name, limit=None, initial_sleep_time=1.1):
    """
    Automates the annotation of hallucinations in model-generated outputs using the Gemini API.

    This function reads input data from a JSONL file and a file containing sampled task IDs. It processes
    tasks that have incorrect generated answers and are included in the sampled IDs. For each task, it
    formats a prompt and sends it to the Gemini model for classification. The results, including the 
    selected error subtype code and rationale, are saved to a specified CSV file.

    Args:
        input_jsonl (str): Path to the JSONL file containing generation outputs.
        sampled_ids_file (str): Path to the .txt file containing sampled task IDs (one ID per line).
        output_csv (str): Path where the resulting CSV file with annotations will be saved.
        model_name (str): The name of the Gemini model to use for annotation.
        limit (int, optional): Maximum number of tasks to process. If None, processes all.
        initial_sleep_time (float, optional): Initial sleep time between retries for rate limiting.

    Returns:
        None
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY environment variable not set.")
        return
    genai.configure(api_key=api_key)

    logging.info(f"Initializing Gemini model: {model_name}")
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model '{model_name}': {e}")
        return

    generation_config = genai.GenerationConfig(
        temperature=0.1, max_output_tokens=300
    )
    safety_settings = []

    input_path = Path(input_jsonl)
    sampled_ids_path = Path(sampled_ids_file)
    output_path = Path(output_csv)

    if not input_path.is_file(): logging.error(f"Input JSONL not found: {input_path}"); return
    if not sampled_ids_path.is_file(): logging.error(f"Sampled IDs file not found: {sampled_ids_path}"); return

    try:
        with open(sampled_ids_path, 'r') as f:
            sampled_ids = {line.strip() for line in f if line.strip()}
        logging.info(f"Loaded {len(sampled_ids)} IDs to annotate from {sampled_ids_path}")
    except Exception as e:
        logging.error(f"Error reading sampled IDs file: {e}"); return

    results = []
    processed_target_count = 0
    total_to_process = len(sampled_ids)

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    task_data = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line {i+1}.")
                    continue

                task_id = str(task_data.get('id', ''))
                is_correct = task_data.get('is_correct', True)

                if task_id in sampled_ids and not is_correct:
                    processed_target_count += 1
                    logging.info(f"Processing task {task_id} ({processed_target_count}/{total_to_process})...")

                    if limit is not None and processed_target_count > limit:
                        logging.info(f"Reached processing limit of {limit}.")
                        break

                    details = {
                        'question': task_data.get('question', ''),
                        'reference_answers': task_data.get('reference_answers', []),
                        'generated_answer': task_data.get('generated_answer', '')
                    }

                    prompt = format_prompt(PROMPT_TEMPLATE, details)
                    if prompt is None:
                        logging.error(f"Task {task_id}: Failed to format prompt. Skipping.")
                        results.append({'id': task_id, 'automated_subtype': '[PROMPT_ERROR]', 'automated_rationale': '[PROMPT_ERROR]', 'full_llm_response': '[PROMPT_ERROR]'})
                        continue

                    current_attempt = 0
                    non_rate_limit_attempts = 0
                    current_sleep = initial_sleep_time
                    while True:
                        current_attempt += 1
                        logging.debug(f"Task {task_id}: Attempt {current_attempt}...")
                        try:
                            response = model.generate_content(
                                prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                            )

                            if not response.candidates:
                                block_reason = "Unknown (No candidates)"
                                if response.prompt_feedback and response.prompt_feedback.block_reason:
                                    block_reason = response.prompt_feedback.block_reason.name
                                logging.warning(f"Task {task_id}: Prompt blocked by API (Reason: {block_reason}). Cannot proceed with this sample.")
                                subtype_code = f"[BLOCKED_PROMPT_{block_reason}]"
                                rationale = f"[BLOCKED_PROMPT_{block_reason}]"
                                response_text = f"[BLOCKED_PROMPT_{block_reason}]"
                                results.append({'id': task_id, 'automated_subtype': subtype_code, 'automated_rationale': rationale, 'full_llm_response': response_text})
                                break

                            candidate = response.candidates[0]
                            if candidate.finish_reason.name != "STOP":
                                logging.warning(f"Task {task_id}: Non-STOP finish reason: {candidate.finish_reason.name}. Content may be incomplete or blocked.")
                                if candidate.finish_reason.name == "SAFETY":
                                     subtype_code = f"[BLOCKED_CONTENT_{candidate.finish_reason.name}]"
                                     rationale = f"[BLOCKED_CONTENT_{candidate.finish_reason.name}]"
                                     response_text = f"[BLOCKED_CONTENT_{candidate.finish_reason.name}] - {candidate.content.parts[0].text if candidate.content.parts else ''}"
                                     results.append({'id': task_id, 'automated_subtype': subtype_code, 'automated_rationale': rationale, 'full_llm_response': response_text})
                                     break 
                                response_text = candidate.content.parts[0].text if candidate.content.parts else "[INCOMPLETE_RESPONSE]"
                            else:
                                response_text = candidate.content.parts[0].text

                            subtype_code, rationale = parse_llm_response(response_text)

                            if subtype_code and rationale:
                                logging.info(f"Task {task_id}: Successfully annotated as {subtype_code}.")
                                results.append({'id': task_id, 'automated_subtype': subtype_code, 'automated_rationale': rationale, 'full_llm_response': response_text})
                                break
                            else:
                                logging.warning(f"Task {task_id}: Parsing failed for response. Retrying API call...")
                                non_rate_limit_attempts += 1
                                if non_rate_limit_attempts >= MAX_NON_RATE_LIMIT_RETRIES:
                                    logging.error(f"Task {task_id}: Exceeded max retries ({MAX_NON_RATE_LIMIT_RETRIES}) for parsing failure. Skipping sample.")
                                    results.append({'id': task_id, 'automated_subtype': '[PARSE_FAILED_MAX_RETRIES]', 'automated_rationale': '[PARSE_FAILED_MAX_RETRIES]', 'full_llm_response': response_text[:500]})
                                    break 
                                time.sleep(min(current_sleep * 1.5, 30))

                        except Exception as e:
                            logging.warning(f"Task {task_id}: API call attempt {current_attempt} failed: {type(e).__name__} - {e}")

                            error_str = str(e).lower()
                            if "rate limit" in error_str or "429" in error_str or "resource has been exhausted" in error_str or "quota" in error_str:
                                wait_time = min(initial_sleep_time * (2 ** (current_attempt % 10)) + np.random.uniform(0, 1), MAX_SLEEP_INTERVAL)
                                logging.info(f"Rate limit/Resource likely hit. Waiting {wait_time:.2f}s...")
                                time.sleep(wait_time)
    
                            else:
                                non_rate_limit_attempts += 1
                                if non_rate_limit_attempts >= MAX_NON_RATE_LIMIT_RETRIES:
                                    logging.error(f"Task {task_id}: Exceeded max retries ({MAX_NON_RATE_LIMIT_RETRIES}) for non-rate-limit error. Skipping sample.")
                                    results.append({'id': task_id, 'automated_subtype': '[API_FAILED_MAX_RETRIES]', 'automated_rationale': f'[API_FAILED_MAX_RETRIES: {type(e).__name__}]', 'full_llm_response': f'[API_FAILED_MAX_RETRIES: {e}]'})
                                    break
                                else:
                                    wait_time = min(initial_sleep_time * (1.5 ** non_rate_limit_attempts), 60)
                                    logging.info(f"Transient error likely. Waiting {wait_time:.2f}s...")
                                    time.sleep(wait_time)
                        finally:
                             if not (subtype_code and rationale) :
                                time.sleep(initial_sleep_time + np.random.uniform(0,0.5))


    except KeyboardInterrupt:
         logging.warning("Keyboard interrupt detected. Saving partial results...")
    except Exception as e:
        logging.error(f"An unexpected error occurred during file processing: {e}", exc_info=True)
    finally:
        if results:
            df_results = pd.DataFrame(results)
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_results.to_csv(output_path, index=False)
                logging.info(f"Saved {len(df_results)} attempted annotations to: {output_path}")
            except IOError as e:
                logging.error(f"Failed to save results to {output_path}: {e}")
        else:
            logging.warning("No results generated to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate hallucination annotation using Gemini API.")
    parser.add_argument("input_jsonl", help="Path to the JSONL file containing generation outputs (e.g., _validation_outputs_for_annotation.jsonl).")
    parser.add_argument("sampled_ids_file", help="Path to the .txt file containing sampled task IDs (one ID per line).")
    parser.add_argument("output_csv", help="Path to save the automated annotations CSV file.")
    parser.add_argument("--model", default="gemini-1.5-flash-latest", help="Gemini model name (e.g., gemini-1.5-flash-latest, gemini-pro).")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to the first N selected samples (for testing).")
    parser.add_argument("--sleep", type=float, default=1.1, help="Initial sleep time between API calls/attempts in seconds.")

    args = parser.parse_args()

    if not Path(args.input_jsonl).is_file():
        print(f"ERROR: Input JSONL file not found at '{args.input_jsonl}'")
        exit(1)
    if not Path(args.sampled_ids_file).is_file():
        print(f"ERROR: Sampled IDs file not found at '{args.sampled_ids_file}'")
        exit(1)
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        exit(1)

    automate_annotations(args.input_jsonl, args.sampled_ids_file, args.output_csv, args.model, args.limit, args.sleep)