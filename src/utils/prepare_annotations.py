import json
import argparse
from pathlib import Path

def add_sampling_flag(original_jsonl_path, sampled_ids_path, output_jsonl_path):
    """
    Reads the original JSONL, adds a sampling flag based on sampled IDs,
    and writes to a new JSONL file.
    """
    original_jsonl_path = Path(original_jsonl_path)
    sampled_ids_path = Path(sampled_ids_path)
    output_jsonl_path = Path(output_jsonl_path)

    if not original_jsonl_path.is_file():
        print(f"Error: Original JSONL file not found at {original_jsonl_path}")
        return
    if not sampled_ids_path.is_file():
        print(f"Error: Sampled IDs file not found at {sampled_ids_path}")
        return

    # Read sampled IDs into a set for fast lookup
    try:
        with open(sampled_ids_path, 'r') as f:
            # Strip whitespace and ignore empty lines
            sampled_ids = {line.strip() for line in f if line.strip()}
        print(f"Read {len(sampled_ids)} sampled IDs from {sampled_ids_path}")
    except Exception as e:
        print(f"Error reading sampled IDs file: {e}")
        return

    output_data = []
    tasks_processed = 0
    tasks_flagged = 0

    try:
        with open(original_jsonl_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    task_data = json.loads(line)
                    tasks_processed += 1
                    task_id = str(task_data.get('id', '')) # Ensure ID is string for comparison

                    # Add the flag
                    if task_id in sampled_ids:
                        task_data['selected_for_annotation'] = True
                        tasks_flagged += 1
                    else:
                        task_data['selected_for_annotation'] = False

                    output_data.append(task_data)

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {original_jsonl_path}")
                except Exception as e:
                    print(f"Warning: Error processing line: {e}")

        print(f"Processed {tasks_processed} tasks from {original_jsonl_path}.")
        print(f"Flagged {tasks_flagged} tasks as selected for annotation.")

        if tasks_flagged != len(sampled_ids):
             print(f"Warning: Number of flagged tasks ({tasks_flagged}) does not match number of sampled IDs ({len(sampled_ids)}). Some IDs might be missing in the original JSONL.")


        # Write the modified data to the output file
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
            for task_data in output_data:
                outfile.write(json.dumps(task_data) + '\n')
        print(f"Successfully wrote flagged data to {output_jsonl_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a sampling flag to Label Studio JSONL data.")
    parser.add_argument("original_jsonl", help="Path to the original JSONL file imported into Label Studio.")
    parser.add_argument("sampled_ids", help="Path to the .txt file containing sampled task IDs (one ID per line).")
    parser.add_argument("output_jsonl", help="Path where the new JSONL file with the flag will be saved.")
    args = parser.parse_args()

    add_sampling_flag(args.original_jsonl, args.sampled_ids, args.output_jsonl)