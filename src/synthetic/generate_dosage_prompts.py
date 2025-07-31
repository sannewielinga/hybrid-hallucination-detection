import json
import random
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MEDICATIONS = {
    "MedA": {
        "rule_type": "weight_linear",
        "params": {"mg_per_kg": 10},
        "units": "mg"
    },
    "MedB": {
        "rule_type": "age_tiered",
        "params": {
            "tiers": [(5, 50.0), (11, 100.0)], 
            "default_dosage_mg": 150.0 
        },
        "units": "mg"
    },
    "MedC": { 
        "rule_type": "weight_fixed_unless_distractor",
        "params": {
            "base_mg_per_kg": 5,
            "distractor_keyword": "renal impairment", 
            "adjustment_factor": 0.5 
        },
        "units": "mg"
    }
}

PATIENT_WEIGHT_RANGES_KG = [(5, 15), (16, 30), (31, 50), (51, 70), (71, 100)]
PATIENT_AGE_RANGES_YEARS = [(1, 4), (5, 11), (12, 17), (18, 60), (61, 80)]

DISTRACTOR_POOL = [
    "The patient also reports a mild headache.",
    "Patient has a known allergy to penicillin (note: this is not the prescribed drug for this question).",
    "The patient's temperature is 37.5°C.",
    "The patient mentioned they have gained 5% of their body weight in the last two months.",
    "The patient has a history of seasonal allergies.",
    "The patient mentions experiencing mild renal impairment.",
    "The patient's sibling also had a similar condition last year."
]

QUESTION_PROMPT_TEMPLATES = [
    "A patient is {age} years old and weighs {weight} kg. They need to be administered {medication_name}. {distractor_sentence_1} {distractor_sentence_2} Considering all information, what is the correct dosage of {medication_name} in {units}? Your answer must be a single number. Output: ",
    "Patient details: Age - {age} years, Weight - {weight} kg. Prescription: {medication_name}. {distractor_sentence_1} {distractor_sentence_2} Based on the provided dosing rules and patient data, calculate the required dosage in {units}. Your answer must be a single number. Output: ",
    "Given a {age}-year-old patient weighing {weight} kg who requires {medication_name}. {distractor_sentence_1} {distractor_sentence_2} Follow the specified dosing instructions to determine the dosage in {units}. Your answer must be a single number. Output: "
]

def get_medication_rule_text(med_name, med_info):
    """
    Return a string explaining the dosing rule for a given medication name and info dict.
    
    Args:
        med_name (str): The name of the medication.
        med_info (dict): The information dict for the medication, containing at least the 'params' key.
            The 'params' dict should contain the dosing parameters specific to the medication.
    
    Returns:
        str: A string explaining the dosing rule.
    """
    if med_name == "MedA":
        return f"The dosing rule for {med_name} is {med_info['params']['mg_per_kg']} {med_info['units']} per kg of body weight."
    elif med_name == "MedB":
        tier_text_parts = []
        for max_age_tier, dosage_tier in med_info['params']['tiers']:
            tier_text_parts.append(f"if age is less than or equal to {max_age_tier} years, give {dosage_tier} {med_info['units']}")
        default_text = f"otherwise (if age is greater than {med_info['params']['tiers'][-1][0]} years), give {med_info['params']['default_dosage_mg']} {med_info['units']}"
        return f"The dosing rule for {med_name} is as follows: {'; '.join(tier_text_parts)}; {default_text}."
    elif med_name == "MedC":
        return (f"The base dosing rule for {med_name} is {med_info['params']['base_mg_per_kg']} {med_info['units']} per kg. "
                f"However, if the patient information indicates '{med_info['params']['distractor_keyword']}', "
                f"the calculated base dosage should be adjusted by multiplying by a factor of {med_info['params']['adjustment_factor']}.")
    return "No specific rule text defined."


def calculate_ground_truth_dosage(med_name, weight_kg, age_years, distractors_for_calc_logic):
    """
    Calculate the appropriate dosage for a given medication based on patient weight, age, and any relevant distractors.

    Args:
        med_name (str): The name of the medication for which the dosage is being calculated.
        weight_kg (float): The weight of the patient in kilograms.
        age_years (int): The age of the patient in years.
        distractors_for_calc_logic (list): List of distractor sentences to determine any special conditions that impact dosage.

    Returns:
        float: The calculated dosage based on the medication's dosing rule, rounded to two decimal places. Returns None if no applicable rule is found.
    """

    rule = MEDICATIONS[med_name]
    if rule["rule_type"] == "weight_linear":
        return round(weight_kg * rule["params"]["mg_per_kg"], 2)
    elif rule["rule_type"] == "age_tiered":
        for max_age, dosage in rule["params"]["tiers"]:
            if age_years <= max_age:
                return float(dosage)
        return float(rule["params"]["default_dosage_mg"])
    elif rule["rule_type"] == "weight_fixed_unless_distractor":
        dosage = round(weight_kg * rule["params"]["base_mg_per_kg"], 2)
        has_relevant_distractor = any(rule["params"]["distractor_keyword"] in d.lower() for d in distractors_for_calc_logic)
        if has_relevant_distractor:
            dosage = round(dosage * rule["params"]["adjustment_factor"], 2)
        return dosage
    return None

def generate_synthetic_data(num_examples_per_medication=50):
    """
    Generate synthetic dosage calculation prompts with distractors, with the corresponding ground truth answers.

    Parameters:
        num_examples_per_medication (int): The number of prompts to generate for each medication.

    Returns:
        list: A list of dicts, where each dict represents a single synthetic prompt with the following fields:
            id (str): The unique identifier for the example.
            question_template_type (str): Always "dosage_calculation".
            medication_name (str): The name of the medication.
            patient_age_years (int): The age of the patient in years.
            patient_weight_kg (float): The weight of the patient in kilograms.
            distractors_in_prompt (list): A list of distractor sentences included in the prompt.
            medication_rule_provided_in_prompt (str): The medication rule text provided in the prompt.
            prompt_text (str): The full text of the prompt given to the language model.
            ground_truth_dosage (float): The ground truth dosage answer for the prompt.
            units (str): The units of the ground truth dosage.
            _debug_distractors_for_calc (list): A list of distractors used for calculation logic (for debugging purposes only).
    """
    dataset = []
    example_id_counter = 0
    for med_name, med_info in MEDICATIONS.items():
        for _ in range(num_examples_per_medication):
            weight = random.randint(min(r[0] for r in PATIENT_WEIGHT_RANGES_KG), 
                                    max(r[1] for r in PATIENT_WEIGHT_RANGES_KG))
            age = random.randint(min(r[0] for r in PATIENT_AGE_RANGES_YEARS), 
                                 max(r[1] for r in PATIENT_AGE_RANGES_YEARS))
            
            num_distractors_to_select = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
            
            available_distractors = list(DISTRACTOR_POOL)
            distractors_for_prompt = []
            distractors_for_calculation_logic = []

            if med_name == "MedC" and med_info["params"].get("distractor_keyword"):
                critical_distractor_text = med_info["params"]["distractor_keyword"]
                if random.random() < 0.6:
                    actual_critical_distractor = next((d for d in DISTRACTOR_POOL if critical_distractor_text in d.lower()), None)
                    if actual_critical_distractor:
                        distractors_for_prompt.append(actual_critical_distractor)
                        distractors_for_calculation_logic.append(actual_critical_distractor)
                        if actual_critical_distractor in available_distractors:
                            available_distractors.remove(actual_critical_distractor)
                        if num_distractors_to_select > 0:
                             num_distractors_to_select -=1
            
            if num_distractors_to_select > 0 and available_distractors:
                chosen_generic_distractors = random.sample(available_distractors, min(num_distractors_to_select, len(available_distractors)))
                distractors_for_prompt.extend(chosen_generic_distractors)
            
            random.shuffle(distractors_for_prompt)

            distractor_sentence_1 = distractors_for_prompt[0] if len(distractors_for_prompt) > 0 else "No additional relevant conditions reported."
            distractor_sentence_2 = distractors_for_prompt[1] if len(distractors_for_prompt) > 1 else ""

            med_rule_text_for_prompt = get_medication_rule_text(med_name, med_info)
            
            question_template_chosen = random.choice(QUESTION_PROMPT_TEMPLATES)
            question_part = question_template_chosen.format(
                age=age,
                weight=weight,
                medication_name=med_name,
                distractor_sentence_1=distractor_sentence_1,
                distractor_sentence_2=distractor_sentence_2.strip(),
                units=med_info["units"]
            )
            
            full_prompt_text_for_llm = f"{med_rule_text_for_prompt} {question_part}".replace("  ", " ").strip()
            
            ground_truth = calculate_ground_truth_dosage(med_name, weight, age, distractors_for_calculation_logic)
            
            dataset.append({
                "id": f"synth_dosage_{example_id_counter:04d}",
                "question_template_type": "dosage_calculation",
                "medication_name": med_name,
                "patient_age_years": age,
                "patient_weight_kg": weight,
                "distractors_in_prompt": distractors_for_prompt,
                "medication_rule_provided_in_prompt": med_rule_text_for_prompt,
                "prompt_text": full_prompt_text_for_llm,
                "ground_truth_dosage": ground_truth,
                "units": med_info["units"],
                "_debug_distractors_for_calc": distractors_for_calculation_logic 
            })
            example_id_counter += 1
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dosage calculation prompts.")
    parser.add_argument("--num_per_med", type=int, default=10, help="Number of examples to generate per medication rule.")
    parser.add_argument("--output_file", type=str, default="synthetic_dosage_prompts.jsonl", help="Output JSONL file.")
    args = parser.parse_args()

    dataset = generate_synthetic_data(num_examples_per_medication=args.num_per_med)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    
    logging.info(f"Generated {len(dataset)} synthetic dosage prompts and saved to {output_path}")

if __name__ == "__main__":
    main()