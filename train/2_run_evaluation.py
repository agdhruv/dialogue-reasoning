import json
from pathlib import Path
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re
import os
from typing import List

# Add project root to path
sys.path.append(str(Path.cwd().parent))

from methods.baselines import cot_answer
from src.clients import get_ollama_client
from src.utils import load_gsm8k_dataset

# --- Constants ---
SPECIAL_TOKENS = {
    "question": "<|question|>",
    "solver": "<|solver|>",
    "critic": "<|critic|>",
    "eot": "<|endofturn|>",
    "eod": "<|endofdialogue|>"
}

MODEL_DIR = Path("lora_debate_ckpts")
ADAPTER_DIR = MODEL_DIR / "checkpoint-357"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OLLAMA_BASE_MODEL = "qwen2.5:1.5b"
RESULTS_DIR = Path("evaluation_results")

# Enable faster matmuls on Ampere+ and optimize inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# --- Model Loading ---
def load_finetuned_model():
    """Loads the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    # Decoder-only models should use LEFT padding for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer

def run_finetuned_model_batch(model, tokenizer, questions: List[str]) -> List[str]:
    """Runs batched inference on the fine-tuned model."""
    prompts: List[str] = []
    for question in questions:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant. Final answer format: {"answer": <number>}.'},
            {'role': 'user', 'content': f'{SPECIAL_TOKENS["question"]} {question}'}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += SPECIAL_TOKENS["solver"]
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eod"])]
        )

    return tokenizer.batch_decode(out, skip_special_tokens=False)

# --- Answer Extraction ---
def extract_answer(text: str) -> str:
    """Extracts the numerical answer from a model's output."""
    # Answer is in JSON format
    pattern = r'\{"answer":\s*([0-9]+(?:\.[0-9]+)?)\}'
    match = re.findall(pattern, text)
    if match:
        return match[-1]
    return None

# --- Main Evaluation Logic ---
def main():
    # Load models and data
    print("Loading models and data...")
    finetuned_model, tokenizer = load_finetuned_model()
    ollama_client = get_ollama_client()
    gsm8k_test = load_gsm8k_dataset("test")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    batch_size = 16

    pending_indices: List[int] = []
    pending_questions: List[str] = []
    pending_gold: List[str] = []

    iterator = tqdm(gsm8k_test.iterrows(), total=len(gsm8k_test), desc="Evaluating models (batched)")
    for index, row in iterator:
        output_file = RESULTS_DIR / f"{index}.json"
        if output_file.exists():
            continue

        question = row["question"]
        gold_answer = str(row["gold_answer"])

        pending_indices.append(index)
        pending_questions.append(question)
        pending_gold.append(gold_answer)

        if len(pending_questions) < batch_size:
            continue

        # Run fine-tuned model on the current batch (GPU-intensive)
        finetuned_outputs = run_finetuned_model_batch(finetuned_model, tokenizer, pending_questions)
        finetuned_answers = [extract_answer(text) for text in finetuned_outputs]

        # For each sample in the batch, run CoT baseline and write results
        for i, idx in enumerate(pending_indices):
            output_file = RESULTS_DIR / f"{idx}.json"
            cot_response = cot_answer(ollama_client, pending_questions[i], OLLAMA_BASE_MODEL)
            cot_output = cot_response.choices[0].message.content
            cot_ans = extract_answer(cot_output)

            result = {
                "question": pending_questions[i],
                "gold_answer": pending_gold[i],
                "cot_output": cot_output,
                "cot_answer": cot_ans,
                "finetuned_output": finetuned_outputs[i],
                "finetuned_answer": finetuned_answers[i],
            }
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

        pending_indices.clear()
        pending_questions.clear()
        pending_gold.clear()

    # Process any remaining samples smaller than batch_size
    if pending_questions:
        finetuned_outputs = run_finetuned_model_batch(finetuned_model, tokenizer, pending_questions)
        finetuned_answers = [extract_answer(text) for text in finetuned_outputs]
        for i, idx in enumerate(pending_indices):
            output_file = RESULTS_DIR / f"{idx}.json"
            cot_response = cot_answer(ollama_client, pending_questions[i], OLLAMA_BASE_MODEL)
            cot_output = cot_response.choices[0].message.content
            cot_ans = extract_answer(cot_output)

            result = {
                "question": pending_questions[i],
                "gold_answer": pending_gold[i],
                "cot_output": cot_output,
                "cot_answer": cot_ans,
                "finetuned_output": finetuned_outputs[i],
                "finetuned_answer": finetuned_answers[i],
            }
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
    