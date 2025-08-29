import json
from pathlib import Path
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

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
ADAPTER_DIR = MODEL_DIR / "checkpoint-600"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OLLAMA_BASE_MODEL = "llama3.1:8b"
RESULTS_DIR = Path("evaluation_results")

# --- Model Loading ---
def load_finetuned_model():
    """Loads the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer

def run_finetuned_model(model, tokenizer, question):
    """Runs inference on the fine-tuned model."""
    messages = [
        {'role': 'system', 'content': f'You are a debate assistant. Respond with alternating segments that start with {SPECIAL_TOKENS["solver"]} or {SPECIAL_TOKENS["critic"]} and finish each turn with {SPECIAL_TOKENS["eot"]}. Use no other tags.'},
        {'role': 'user', 'content': f'{SPECIAL_TOKENS["question"]} {question}'}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += SPECIAL_TOKENS["solver"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eod"])]
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)

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

    for index, row in tqdm(gsm8k_test.iterrows(), total=len(gsm8k_test), desc="Evaluating models"):
        
        output_file = RESULTS_DIR / f"{index}.json"
        if output_file.exists():
            continue
        
        question = row["question"]
        gold_answer = str(row["gold_answer"])

        # Run CoT model
        cot_response = cot_answer(ollama_client, question, OLLAMA_BASE_MODEL)
        cot_output = cot_response.choices[0].message.content
        cot_answer_extracted = extract_answer(cot_output)
        
        # Run fine-tuned model
        finetuned_output = run_finetuned_model(finetuned_model, tokenizer, question)
        finetuned_answer_extracted = extract_answer(finetuned_output)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "cot_output": cot_output,
            "cot_answer": cot_answer_extracted,
            "finetuned_output": finetuned_output,
            "finetuned_answer": finetuned_answer_extracted,
        }
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
    