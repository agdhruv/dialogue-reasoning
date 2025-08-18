"""
Convert raw debate traces into flat text suitable for causal-LM fine-tuning.
This script uses the Hugging Face `datasets` library for robust processing.

Expected input JSONL schema (one object per line):
{
  "question": str,
  "gold_answer": int | str,
  "trace_type": "FIXED_BY_DEBATE" | "ALREADY_CORRECT" | "INCORRECT",
  "turns": [
      { "speaker": "Solver" | "Critic",
        "receiver": "Solver" | "Critic",
        "content": str,
        "raw_response": dict | null
      }
  ]
}

Output:
  <out_dir>/train.txt
  <out_dir>/val.txt
  <out_dir>/stats.json
"""
import json
import math
from pathlib import Path
from typing import List
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "question": "<|question|>",
    "solver": "<|solver|>",
    "critic": "<|critic|>",
    "eot": "<|endofturn|>",
    "eod": "<|endofdialogue|>"
}

def serialize_dialogue(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Serialize turns and append TERMINATE line at the end.
    Return the serialized text or None if the trace is incorrect.
    """
    messages = [
        {'role': 'system', 'content': f'You are a debate assistant. Respond with alternating segments that start with <|solver|> or <|critic|> and finish each turn with <|endofturn|>. Use no other tags.'},
        {'role': 'user', 'content': f'{SPECIAL_TOKENS["question"]} {example["question"].strip()}'},
    ]
    assistant_response = ''
    for t in example["turns"][1:]: # skipping the first turn, since that's the critic asking the solver the question
        role_tag = SPECIAL_TOKENS["solver"] if t["speaker"] == "Solver" else SPECIAL_TOKENS["critic"]
        assistant_response += f'{role_tag} {t["content"].strip()}{SPECIAL_TOKENS["eot"]}'
    assistant_response += f'{SPECIAL_TOKENS["eod"]}'
    messages.append({'role': 'assistant', 'content': assistant_response})

    serialized = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": serialized, 'trace_type': example['trace_type']}

def write_dataset_to_text(dataset: Dataset, path: Path, column: str) -> None:
    """Writes a single column from a dataset to a text file, one line per entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for example in dataset:
            f.write(json.dumps({"text": example[column]}) + "\n")

def main() -> None:
    input_jsonl = Path("generated_data/debate_traces.jsonl") # Input JSONL file
    out_dir = Path("generated_data/") # Output directory
    val_ratio = 0.05 # Fraction for validation split
    seed = 42 # RNG seed
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    
    # 1. Load jsonl as HF dataset
    print(f"► Loading dataset from {input_jsonl}...")
    ds = load_dataset("json", data_files=str(input_jsonl), split="train")

    # 2. Filter out incorrect traces
    initial_count = len(ds)
    ds = ds.filter(lambda x: x["trace_type"] != "INCORRECT")
    print(f"• Removed {initial_count - len(ds):,} incorrect traces. {len(ds):,} remaining.")

    # 3. Serialize dialogues
    print("► Serializing dialogues...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = ds.map(serialize_dialogue, remove_columns=ds.column_names, fn_kwargs={"tokenizer": tokenizer})

    # 4. Oversample FIXED_BY_DEBATE examples
    print("► Oversampling FIXED_BY_DEBATE examples...")
    fixed_by_debate_ds = ds.filter(lambda x: x['trace_type'] == "FIXED_BY_DEBATE")
    already_correct_count = len(ds.filter(lambda x: x['trace_type'] == "ALREADY_CORRECT"))
    
    balanced_ds = ds
    if len(fixed_by_debate_ds) < already_correct_count:
        multiplier = math.ceil(already_correct_count / len(fixed_by_debate_ds))
        oversampled_fixed_ds = fixed_by_debate_ds.repeat(multiplier - 1)
        balanced_ds = concatenate_datasets([ds, oversampled_fixed_ds])
        print(f"• Oversampling 'FIXED_BY_DEBATE' traces by a factor of {multiplier}. Adding {len(oversampled_fixed_ds):,} examples.")
    
    balanced_ds = balanced_ds.shuffle(seed=seed)
    print(f"• After oversampling: {len(balanced_ds):,} total sequences")

    # 5. Split into train and val
    split_ds = balanced_ds.train_test_split(test_size=val_ratio, seed=seed)
    train_ds, val_ds = split_ds['train'], split_ds['test']
    print(f"• Train / Val split: {len(train_ds):,} / {len(val_ds):,}")

    # 6. Write to text files
    write_dataset_to_text(train_ds, out_dir / "train.txt", column="text")
    write_dataset_to_text(val_ds, out_dir / "val.txt", column="text")

    # 7. Write stats
    stats = {
        "total_raw": initial_count,
        "total_after_filtering": len(ds),
        "total_after_oversample": len(balanced_ds),
        "train": len(train_ds),
        "val": len(val_ds),
        "special_tokens": SPECIAL_TOKENS,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("✓ Done.  Files written to", out_dir.resolve())

if __name__ == "__main__":
    main()
