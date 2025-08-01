import sys
import os
import textwrap

PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in analyzing and classifying errors (hallucinations) made by other language models in the medical domain. Your goal is to provide the **MOST specific and accurate classification** based on the provided taxonomy and a rigorous step-by-step process.

**Taxonomy:**

**1. Factuality Errors:** The generated content contradicts established, verifiable real-world medical knowledge or facts *external* to the information implicitly or explicitly provided in the `question`.
    *   **A1_DosageAdministration:** Incorrect medical dosage, frequency, route, procedure.
    *   **A2_ContraindicationIndication:** Wrongly states safe/unsafe or indicated/contraindicated for the subject of the question.
    *   **A3_DiagnosticCriteriaDefinition:** Misrepresents established medical criteria, definitions, symptoms, timelines, or the established function/purpose of a medical concept/method **relevant to the question**.
    *   **B1_StatisticalEpidemiological:** Fabricates/misstates medical stats, efficacy, risk factors related to the question's topic.
    *   **B2_FabricatedEntityGuideline:** Invents non-existent medical entities, guidelines, studies related to the question's topic.
    *   **Factuality_Other:** Other factual error based on external world knowledge, not covered by A1-B2. **(RESERVED: Use ONLY if you can explicitly justify why NONE of A1-B2 apply).**

**2. Faithfulness Errors:** The generated content is inconsistent with the implicit or explicit information, constraints, or topic defined **within the `question` itself**. It ignores, contradicts, or misuses elements of the query.
    *   **A1_ContextIgnorant (Question-Based):** Overlooks or contradicts key entities, relationships, or constraints *mentioned or implied within the question*. (e.g., answering about the wrong drug when the question specified one).
    *   **A2_InstructionMisinterpretation:** Fails to follow the specific *task* or *format* instructed by the question (e.g., providing an explanation when asked to list, giving a single item when asked for multiple).
    *   **B1_ExtrapolationAddition (Question-Based):** Adds significant details or makes claims about the subject of the question that are not directly addressed or reasonably inferable *from the scope defined by the question*, even if factually plausible in a broader sense.
    *   **Faithfulness_Other:** Other error unfaithful to the question's explicit or implicit constraints. **(RESERVED: Use ONLY if you can explicitly justify why NONE of A1-B1 apply).**

**3. Other/Unclear:** The error type is ambiguous, a mix of types, or doesn't fit the above categories clearly.

**--- Analysis Task ---**

**Instructions:** Follow these steps precisely:
1.  Analyze the 'Incorrect Generated Answer' considering the 'Question' and the 'Correct/Reference Answer(s)'. Treat the **question itself as the primary source of context and constraints**.
2.  **Step 1: Determine Primary Error Type.**
    *   First, evaluate **Faithfulness**: Does the answer directly address the entities, constraints, and task defined *by the question*? Does it ignore parts of the question (A1)? Does it fail the *task* requested (A2)? Does it go beyond the question's scope (B1)?
    *   If the answer *is* faithful to the question's topic and constraints but contains incorrect information based on external knowledge, evaluate **Factuality**.
    *   If neither fits well, consider **Other/Unclear**.
3.  **Step 2: Consider Specific Subtypes.**
    *   If **Factuality**: Methodically check A1-B2 based on external knowledge related to the question's subject.
    *   If **Faithfulness**: Methodically check A1 (ignoring question elements), A2 (wrong task/format), or B1 (adding irrelevant details beyond question scope).
4.  **Step 3: Select Final Code.**
    *   Choose the **single most specific A/B code** identified in Step 2 that accurately describes the core error mechanism.
    *   **ONLY IF NONE** of the specific A/B codes for the chosen primary type accurately fit the error, select the corresponding '_Other' code (Factuality_Other or Faithfulness_Other).
    *   If the primary type was Other/Unclear, select that.
5.  **Step 4: Provide Rationale.**
    *   Justify your chosen code by explaining how the error violates external facts (Factuality) OR fails to adhere to the question's implicit/explicit constraints (Faithfulness).
    *   **If you selected an '_Other' code, you MUST explicitly state why EACH of the specific A/B codes for that primary type were considered unsuitable. Keep rationale brief.**

**Case Details:**

*   **Question:** {question}
*   **Correct/Reference Answer(s):** {reference_answers}
*   **Incorrect Generated Answer:** {generated_answer}

**--- Classification Output ---**

**Selected Subtype Code:** [Provide the SINGLE specific code OR Other_Unclear]
**Rationale:** [Follow instructions in Step 4 above.]
"""

USE_LOCAL_MODEL = False
LOCAL_MODEL_NAME = "cognitivecomputations/TinyDolphin-2.8-1.1b" 
MAX_NEW_TOKENS = 250

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_multiline_input(prompt):
    """Gets multiline input from the user."""
    print(f"{prompt} (Press Enter twice to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except EOFError:
             break
    return "\n".join(lines)

def format_prompt(template, details):
    """Formats the prompt template with the provided details."""
    try:
        return template.format(**details)
    except KeyError as e:
        print(f"\n--- ERROR: Missing placeholder in template: {e} ---")
        print("--- Please ensure your PROMPT_TEMPLATE has placeholders for: ---")
        print("--- question, context, reference_answers, generated_answer ---")
        sys.exit(1)

# --- Main Loop ---

if USE_LOCAL_MODEL:
    print("Attempting to load local model...")
    try:
        from transformers import pipeline, AutoTokenizer
        import torch
        print(f"Loading tokenizer: {LOCAL_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        print(f"Loading pipeline for: {LOCAL_MODEL_NAME}")
        pipe = pipeline(
            "text-generation",
            model=LOCAL_MODEL_NAME,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        print("Local model loaded successfully.")
        local_model_loaded = True
    except ImportError:
        print("\n--- WARNING: `transformers` library not found. ---")
        print("--- Please install it (`pip install transformers torch accelerate`) ---")
        print("--- Continuing without local model integration. ---")
        local_model_loaded = False
        USE_LOCAL_MODEL = False
    except Exception as e:
        print(f"\n--- ERROR loading local model {LOCAL_MODEL_NAME}: {e} ---")
        print("--- Check model name/path and dependencies. ---")
        print("--- Continuing without local model integration. ---")
        local_model_loaded = False
        USE_LOCAL_MODEL = False

while True:
    clear_screen()
    print("--- Enter Case Details (Type 'quit' at any prompt to exit) ---")

    details = {}
    details['question'] = input("Question: ")
    if details['question'].lower() == 'quit': break

    details['reference_answers'] = get_multiline_input("Correct/Reference Answer(s)")
    if details['reference_answers'].lower().strip() == 'quit': break

    details['generated_answer'] = get_multiline_input("Incorrect Generated Answer")
    if details['generated_answer'].lower().strip() == 'quit': break

    full_prompt = format_prompt(PROMPT_TEMPLATE, details)

    print("\n" + "="*80)
    print("--- Generated Prompt (Copy this and paste into your LLM) ---")
    print("="*80)
    print(full_prompt)
    print("="*80 + "\n")

    if USE_LOCAL_MODEL and local_model_loaded:
        print(f"--- Attempting classification with local model: {LOCAL_MODEL_NAME} ---")
        print("--- Please wait... ---")
        try:
            outputs = pipe(
                full_prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id
            )
            llm_response = outputs[0]['generated_text']
            print("\n" + "-"*80)
            print("--- Local LLM Response ---")
            print("-"*80)
            print(textwrap.fill(llm_response, width=80))
            print("-"*80 + "\n")
            print("--- Please review this suggestion carefully before using it. ---")

        except Exception as e:
            print(f"\n--- ERROR during local model generation: {e} ---")

    input("\n--- Press Enter to continue with the next case, or Ctrl+C to exit ---")

print("\nExiting.")
