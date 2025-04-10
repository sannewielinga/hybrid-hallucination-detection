<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** Replace "github_username", "repo_name" etc. with your actual repository details.
*** Remove or comment out any badges you do not want to display.
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<!-- <br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">A Hybrid Framework for Hallucination Detection in Smaller LLMs</h3>

  <p align="center">
    This repository contains the code and experiments for the Master's thesis investigating a hybrid framework for detecting hallucinations (factuality and faithfulness errors) in smaller Large Language Models (LLMs), specifically focusing on the medical question-answering domain.
    <br />
    <!-- <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a> -->
    <br />
    <br />
    <!-- <a href="https://github.com/github_username/repo_name">View Demo</a> -->
    <!-- · -->
    <!-- <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a> -->
    <!-- · -->
    <!-- <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a> -->
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#setup">Setup</a>
        <ul>
            <li><a href="#prerequisites">Prerequisites</a></li>
            <li><a href="#local-setup">Local Setup</a></li>
            <li><a href="#docker-setup">Docker Setup</a></li>
        </ul>
    </li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#running-experiments">Running Experiments</a>
        <ul>
            <li><a href="#overview">Overview</a></li>
            <li><a href="#master-script-usage">Master Script Usage</a></li>
            <li><a href="#running-with-docker">Running with Docker</a></li>
        </ul>
    </li>
    <li><a href="#workflow--data">Workflow & Data</a>
        <ul>
            <li><a href="#experimental-stages">Experimental Stages</a></li>
            <li><a href="#input-data-requirements">Input Data Requirements</a></li>
            <li><a href="#annotation-workflow">Annotation Workflow</a></li>
            <li><a href="#output-structure">Output Structure</a></li>
        </ul>
    </li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the code and experiments for the Master's thesis investigating a hybrid framework for detecting hallucinations (factuality and faithfulness errors) in smaller LLMs (1B parameter range), specifically focusing on the medical question-answering domain.

The primary goal is to evaluate and potentially combine two complementary approaches:

1. **Semantic Entropy (SE)**
    Measures global uncertainty and semantic divergence across multiple model generations for a given prompt. Based on the work by Farquhar et al. (2024).

2. **Internal Signal (IS) Probing**
    An exploratory approach that trains a simple classifier on the LLM's internal hidden states (specifically, the last hidden state of the last generated token in the low-temperature output) to predict output correctness (`True`/`False`), aiming to capture signals potentially missed by SE.

The hypothesis is that combining these signals (e.g., through weighted averaging or a meta-classifier, although only simple averaging is implemented here) could lead to more robust hallucination detection than either method alone, especially for resource-constrained deployments common in healthcare.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REPOSITORY STRUCTURE -->
## Repository Structure

```text
hybrid-hallucination-detection/
├── configs/
│ ├── bioasq_llama3_jpv5oxug.yaml
│ └── ... (other configs)
│
├── data/ 
│ └── bioasq/
│ └── training11b.json 
│
├── annotation_data/ 
│ └── jpv5oxug_annotations_processed.csv
│ └── ...
│
├── src/
│ ├── semantic_uncertainty/ 
│ │ ├── generate.py
│ │ └── compute_se.py
│ │
│ ├── internal_signals/
│ │ └── probe.py
│ │
│ ├── preparation/
│ │ └── create_analysis_dataset.py
│ │
│ ├── analysis/ 
│ │ └── evaluate_subtypes.py 
│ │
│ ├── utils/ 
│ │ ├── init.py
│ │ ├── config_loader.py
│ │ ├── data_utils.py
│ │ ├── eval_utils.py
│ │ ├── logging_utils.py
│ │ ├── models.py
│ │ ├── p_true.py
│ │ └── semantic_entropy.py
│ │ └── utils.py 
│ │
│ └── run_experiment.py
│
├── outputs/
│ └── runs/
│ └── <run_id>/ 
│ ├── logs/ 
│ ├── plots/
│ ├── validation_generations.pkl 
│ ├── uncertainty_measures.pkl 
│ ├── <run_id>_internal_signal_results.csv 
│ └── <run_id>_final_analysis_data.csv 
│ └── ...
│
├── Dockerfile
├── requirements.txt 
├── .dockerignore 
├── .gitignore 
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SETUP -->
## Setup

### Prerequisites

- Python 3.10
- `pip` package installer
- Git
- **(For GPU Usage)** NVIDIA GPU with CUDA toolkit compatible with the PyTorch version in `requirements.txt` (e.g., CUDA 11.8+ or 12.1+). Ensure NVIDIA drivers are installed.
- **(For Docker GPU Usage)** Docker installed, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host machine.

### Local Setup

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd hybrid-hallucination-detection
    ```

2. **Create and activate a virtual environment (Recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .\ .venv\Scripts\activate  # On Windows PowerShell
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Hugging Face Login:** If using gated models (like Llama 3), log in via the terminal:

    ```bash
    huggingface-cli login
    ```

    You'll need a Hugging Face account and an access token with read permissions. Make sure you have also accepted the terms for the specific models on the Hugging Face Hub website.
5. **Place Data:** Ensure raw dataset files (like BioASQ's `training11b.json`) are placed in the correct location under the `data/` directory (e.g., `data/bioasq/training11b.json`) as expected by `src/utils/data_utils.py`.

### Docker Setup

Using Docker ensures a consistent environment across different machines.

1. **Install Prerequisites:** Docker and NVIDIA Container Toolkit (for GPU support).
2. **Build the Docker image:** From the project root directory (`hybrid-hallucination-detection/`):

    ```bash
    docker build -t hallucination-detector .
    ```

    This command builds an image named `hallucination-detector` based on the `Dockerfile`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONFIGURATION -->
## Configuration

Experiments are defined and controlled using YAML files located in the `configs/` directory. Each file represents a single experimental run and defines parameters for all pipeline stages.

Key parameters include:

- `run_id`: Unique identifier for the run (used for output directories and potentially linking data).
- `stages`: Boolean flags (`true`/`false`) to enable/disable specific pipeline stages for a given execution.
- `dataset`: Name of the dataset (e.g., `bioasq`).
- `model_name`: Hugging Face Hub identifier or local path for the LLM.
- Generation parameters (`num_few_shot`, `temperature`, `top_p`, `num_generations`, etc.).
- SE parameters (`entailment_model`, `strict_entailment`, etc.).
- IS probe parameters (`probe_test_size`, `probe_seed`, `probe_accuracy_threshold`).
- Analysis parameters (`analysis_score_columns`, etc.).
- `processed_annotation_dir`: Directory containing the processed annotation CSV files needed for the `prepare_analysis` stage.

Refer to the example configs (e.g., `configs/bioasq_llama3_jpv5oxug.yaml`) and the `argparse` setup within each stage's script for a full list of configurable parameters.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATA SOURCES -->

## Input Data Requirements

- **Raw Datasets:** Place original dataset files in the `data/` directory (e.g., `data/bioasq/training11b.json`). The script `src/utils/data_utils.py` handles loading logic for supported datasets ("bioasq", "medquad", etc.). Adapt this script if using different dataset formats or locations.
- **Processed Annotations:** This is crucial input for the `prepare_analysis` stage. After performing manual annotation in Label Studio and processing the export (see [Annotation Workflow](#annotation-workflow)), the resulting CSV file must be placed in the directory specified by `processed_annotation_dir` in the YAML config (default is `annotation_data/`) and named precisely `<run_id>_annotations_processed.csv`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RUNNING THE PIPELINE -->

## Running Experiments

### Overview

The experimental workflow is divided into distinct stages managed by the master script `src/run_experiment.py`. This script reads a YAML configuration file and executes the enabled stages sequentially via subprocess calls. Each stage reads input files (usually from the previous stage's output) and writes its results to a dedicated directory within `outputs/runs/<run_id>/`.

### Master Script Usage

To run experiments, use `src/run_experiment.py`:

```bash
python src/run_experiment.py <path_to_config_yaml> [--stages <stage_name_1> <stage_name_2> ...]
```

- `<path_to_config_yaml>`: Relative path to the YAML configuration file for the desired run (e.g., configs/bioasq_llama3_jpv5oxug.yaml).
- `--stages` (Optional): Specify which pipeline stage(s) to execute. If omitted, all stages marked true in the YAML config are run. Available stages:
    - `generate`
    - `compute_se`
    - `probe_is`
    - `prepare_analysis` (Requires corresponding processed annotation file)
    - `evaluate_subtypes` (Requires output from prepare_analysis)


### Examples

#### Run the full pipeline defined in the config

```bash
python src/run_experiment.py configs/bioasq_llama3_jpv5oxug.yaml
```

#### Run only generation and SE computation

```bash
python src/run_experiment.py configs/bioasq_llama3_jpv5oxug.yaml --stages generate compute_se
```

#### Run only the final analysis (assuming previous stages are complete and annotations exist)

```bash
python src/run_experiment.py configs/bioasq_llama3_jpv5oxug.yaml --stages evaluate_subtypes
```

### Running with Docker

1. Build the image: `docker build -t hallucination-detector .`

2. Run the container: Mount necessary host directories in the container.

```bash
docker run \
  --gpus all \
  --rm -it \
  -v "$(pwd)/configs":/app/configs \
  -v "$(pwd)/data":/app/data \
  -v "$(pwd)/annotation_data":/app/annotation_data \
  -v "$(pwd)/outputs":/app/outputs \
  hallucination-detector \
  configs/<your_config_file>.yaml [--stages <stage_name_1> ...]
```

- Adjust host paths if needed.
- Remove --gpus all for CPU-only execution (not advised!).


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- WORKFLOW & DATA -->
## Workflow & Data

### Experimental Stages

The pipeline defined in `run_experiment.py` consists of the following stages:

1. **generate** (`src/semantic_uncertainty/generate.py`): Loads data, initializes LLM, generates low-T and high-T answers, calculates P(True) if enabled. Outputs .pkl files with generations/embeddings and a .jsonl file for annotation.
2. **compute_se** (`src/semantic_uncertainty/compute_se.py`): Loads generation data, loads NLI model, calculates SE, Naive Entropy, etc. Outputs uncertainty_measures.pkl.
3. **probe_is** (`src/internal_signals/probe.py`): Loads generation data, trains/evaluates Logistic Regression probe on hidden states, calculates simple hybrid scores. Outputs_internal_signal_results.csv.
4. **(Manual Step) Annotation**: See [Annotation Workflow](#annotation-workflow).
5. **prepare_analysis** (`src/preparation/create_analysis_dataset.py`): Loads processed annotations, all uncertainty scores (SE etc.), and IS probe results. Merges them into a single CSV file. Outputs_final_analysis_data.csv.
6. **evaluate_subtypes** (`src/analysis/evaluate_subtypes.py`): Loads the final merged data. Performs statistical analysis and generates plots comparing specified score columns across hallucination types/subtypes. Outputs plots to plots/ directory.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ANNOTATION WORKFLOW -->
## Annotation Workflow

Annotation is the manual step required between automated stages.

1. Run the generate stage. This creates `<run_id>_validation_outputs_for_annotation.jsonl` in the run's output directory.

2. Use the `stratified_sampler.py` script with the run's `validation_generations.pkl` and `uncertainty_measures.pkl` to select IDs based on SE scores, saving them to `sampled_ids_for_annotation.txt`.

3. Use the `add_sampling_flag.py` script (needs adaptation or creation) to process the full `<run_id>_validation_outputs_for_annotation.jsonl` and the ~sampled_ids_for_annotation.txt~.
This creates a new JSONL file (e.g., `<run_id>_subset_for_annotation.jsonl`) containing all original samples but with an added boolean flag (selected_for_annotation) indicating which ones were sampled.

4. Import the flagged JSONL (`<run_id>_subset_for_annotation.jsonl`) into a Label Studio project configured with the appropriate labeling interface (XML provided previously). Filter the Label Studio view to show only tasks where selected_for_annotation is True. Perform the manual annotation, assigning hallucination_type and hallucination_subtype.

5. Export the completed annotations from Label Studio as CSV (e.g., `<run_id>_annotations_raw.csv`). Ensure the processing logic within src/preparation/create_analysis_dataset.py correctly infers types and merges subtype columns from this raw export. Run the prepare_analysis stage, which uses the raw CSV export and saves the processed data as `outputs/runs/<run_id>/<run_id>_final_analysis_data.csv`. Ensure the raw_annotations_csv path points to your actual Label Studio export.

6. Execute the evaluate_subtypes stage using `run_experiment.py`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OUTPUT STRUCTURE -->
## Output Structure

All generated files for a run are stored within `outputs/runs/<run_id>/`.

- `validation_generations.pkl`: Raw model outputs, embeddings.
- `uncertainty_measures.pkl`: SE, Naive Entropy, P(True) scores for all validation samples.
- `<run_id>_internal_signal_results.csv`: IS probe scores & hybrid scores (for IS test split samples).
- `<run_id>_final_analysis_data.csv`: The final merged dataset containing processed annotations and all calculated scores for the annotated subset. Used as  input for evaluate_subtypes.
- `plots/`: Directory containing analysis plots.
- `<run_id>_pipeline.log`: Master log file for the run.
- `<run_id>_validation_outputs_for_annotation.jsonl`: Intermediate file generated for annotation import.
- `experiment_details.pkl`: Saved configuration and details from the generation stage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- NOTES AND FUTURE WORK -->
## Notes and Future Work

### Month 1 (Deadline: 28-04-2025)

- [ ] Research questions
- [ ] Literature review
- [x] Data sources (BioASQ and MedQuAD)
- [x] Basic structure of thesis report
- [x] Implement baseline models
- [x] Implement semantic entropy
- [ ] Thesis writing

### Month 2 (Deadline: 28-05-2025)

- [ ] Optimize semantic entropy
- [x] First experiments on data
- [x] Implement internal-signal module
- [ ] Thesis writing

### Month 3 (Deadline: 28-06-2025)

- [ ] Implement hybrid pipeline
- [ ] Experiment with different ensembling methods
- [ ] Combine and improve implementation (including both separate modules)
- [ ] Thesis writing

### Month 4 (Deadline: 28-07-2025)

- [ ] Performance evaluation
- [ ] Thesis writing

### Month 5 (Deadline: 28-08-2025)

- [ ] Final writing
- [ ] Presentation

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Sanne Wielinga - [LinkedIn](https://www.linkedin.com/in/sanne-wielinga-501914114/) - [sannwielinga@gmail.com](sannwielinga@gmail.com)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- Farquhar et al. (2024) for the Semantic Entropy concept.
- Orgad et al. (2024) & Su et al. (2024) for inspiration on internal signal analysis.
- Hugging Face 🤗 for the transformers library, models, and datasets.
- Label Studio for the annotation interface.
- My thesis supervisors: Prof. Dr Natasha Alechina & dr.ir. Clara Maathuis.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sannewielinga/hybrid-hallucination-detection.svg?style=for-the-badge
[contributors-url]: https://github.com/sannewielinga/hybrid-hallucination-detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sannewielinga/hybrid-hallucination-detection.svg?style=for-the-badge
[forks-url]: https://github.com/sannewielinga/hybrid-hallucination-detection/network/members
[stars-shield]: https://img.shields.io/github/stars/sannewielinga/hybrid-hallucination-detection.svg?style=for-the-badge
[stars-url]: https://github.com/sannewielinga/hybrid-hallucination-detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/sannewielinga/hybrid-hallucination-detection.svg?style=for-the-badge
[issues-url]: https://github.com/sannewielinga/hybrid-hallucination-detection/issues
[license-shield]: https://img.shields.io/github/license/sannewielinga/hybrid-hallucination-detection.svg?style=for-the-badge
[license-url]: https://github.com/sannewielinga/hybrid-hallucination-detection/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: <https://linkedin.com/in/linkedin_username>

