import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from methods.baselines import zero_shot_answer, cot_answer
from methods.debate import debate
from src.clients import get_azure_openai_client, get_ollama_client
from src.utils import load_gsm8k_dataset

def process_question(index: int, row: pd.Series, results_dir: Path, solver_client: OpenAI | AzureOpenAI, solver_model: str, critic_client: AzureOpenAI, critic_model: str):
    """
    Runs all methods for a single question and saves results by looping through a methods dictionary.
    """
    question = row["question"]

    methods = {
        "zero_shot": {
            "runner": lambda: zero_shot_answer(solver_client, question, solver_model),
            "serializer": lambda response: response.model_dump()
        },
        "cot": {
            "runner": lambda: cot_answer(solver_client, question, solver_model),
            "serializer": lambda response: response.model_dump()
        },
        "debate": {
            "runner": lambda: debate(question, solver_client=solver_client, solver_model=solver_model, critic_client=critic_client, critic_model=critic_model),
            "serializer": lambda response: response.to_list()
        }
    }

    for method_name, config in methods.items():
        method_dir = results_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        output_file = method_dir / f"{index}.json"

        if not output_file.exists():
            response = config["runner"]()
            serialized_data = config["serializer"](response)
            with open(output_file, "w") as f:
                json.dump(serialized_data, f, indent=4)

if __name__ == "__main__":
    experiments = [
        {
            "solver_client_type": "ollama",
            "solver_model": "llama3",
        },
    ]

    critic_client = get_azure_openai_client()
    
    print("Loading GSM8K dataset...")
    dataset_split = "test"
    top_k = 100
    gsm_df = load_gsm8k_dataset(dataset_split, top_k)

    for exp in experiments:
        experiment_id = f"{dataset_split}_{exp['solver_client_type']}_{exp['solver_model']}"
        print(f"--- Running Experiment: {experiment_id} ---")
        
        # Set up client for the current experiment
        if exp['solver_client_type'] == 'ollama':
            solver_client = get_ollama_client()
        else: # azure
            solver_client = get_azure_openai_client()

        # 2. Set up results directory
        results_dir = Path(f"results/gsm8k/{experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Process questions
        print(f"Processing {len(gsm_df)} questions...")
        for index, row in tqdm(gsm_df.iterrows(), total=len(gsm_df), desc=f"Processing GSM8K ({exp['solver_model']})"):
            process_question(index, row, results_dir, solver_client, exp['solver_model'], critic_client, exp['critic_model'])

    # idx = 7
    # conversation = debate(gsm_df.loc[idx]['question'], MODEL_NAME, client)
    # print(f"Question {idx}: {gsm_df.loc[idx]['question']}")
    # print(f"Gold answer: {gsm_df.loc[idx]['gold_answer']}")
    # conversation.print()