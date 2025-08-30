from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path
import os
import sys
sys.path.append(str(Path.cwd().parent))
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from methods.baselines import cot_answer
from methods.debate import debate
from src.clients import get_ollama_client, get_openai_client
from src.utils import load_gsm8k_dataset
import multiprocessing as mp

def process_question(index: int, row: pd.Series, results_dir: Path, solver_client: OpenAI | AzureOpenAI, solver_model: str, critic_client: AzureOpenAI, critic_model: str):
    """
    Runs all methods for a single question and saves results by looping through a methods dictionary.
    """
    question = row["question"]

    methods = {
        "cot": {
            "runner": lambda: cot_answer(solver_client, question, solver_model),
            "serializer": lambda response: response.model_dump()
        },
        "debate": {
            "runner": lambda: debate(question, solver_client=solver_client, solver_model=solver_model, critic_client=critic_client, critic_model=critic_model, max_rounds=5),
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

def process_chunk(chunk_data):
    """
    Process a chunk of questions. This function is designed to be called by multiprocessing.
    """
    chunk_df, results_dir, solver_client_type, solver_model, critic_model = chunk_data
    
    # Create clients inside the worker process
    if solver_client_type == 'ollama':
        solver_client = get_ollama_client()
    else:  # azure
        solver_client = get_openai_client()
    
    critic_client = get_openai_client()
    
    for index, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Processing GSM8K ({solver_model})"):
        process_question(index, row, results_dir, solver_client, solver_model, critic_client, critic_model)

def test_function(idx: int, gsm_df: pd.DataFrame, solver_model: str, solver_client: OpenAI | AzureOpenAI, critic_model: str, critic_client: AzureOpenAI):
    conversation = debate(gsm_df.loc[idx]['question'], solver_client=solver_client, solver_model=solver_model, critic_client=critic_client, critic_model=critic_model)
    print(f"Question {idx}: {gsm_df.loc[idx]['question']}")
    print(f"Gold answer: {gsm_df.loc[idx]['gold_answer']}")
    conversation.print()

if __name__ == "__main__":
    experiments = [
        {
            "solver_client_type": "ollama",
            "solver_model": "qwen2.5:1.5b",
        }
    ]
    critic_model = "gpt-4o"
    
    print("Loading GSM8K dataset...")
    dataset_split = "train"
    top_k = None
    gsm_df = load_gsm8k_dataset(dataset_split, top_k)
    
    # test_function(23, gsm_df, "qwen2.5:1.5b", get_ollama_client(), "gpt-4o", get_openai_client())

    for exp in experiments:
        experiment_id = f"{dataset_split}_{exp['solver_client_type']}_{exp['solver_model']}"
        print(f"--- Running Experiment: {experiment_id} ---")

        # Set up results directory
        results_dir = Path(f"results/gsm8k/{experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Process questions
        print(f"Processing {len(gsm_df)} questions...")
        
        # Split dataframe into chunks for parallel processing
        num_workers = os.cpu_count() // 2
        chunk_size = len(gsm_df) // num_workers
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(gsm_df)
            chunk_df = gsm_df.iloc[start_idx:end_idx].copy()
            chunks.append((chunk_df, results_dir, exp['solver_client_type'], exp['solver_model'], critic_model))
        
        # Process chunks in parallel
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc=f"Processing GSM8K ({exp['solver_model']}) with {num_workers} workers"
            ))