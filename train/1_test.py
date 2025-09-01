from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from pathlib import Path
import torch
from tqdm import tqdm
from peft import PeftModel

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

# Load tokenizer from the adapter directory
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Resize the token embeddings to match the new tokenizer
model.resize_token_embeddings(len(tokenizer))

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

model.eval()

# --- 1. Test on a simple example ---
print("--- Testing on a simple example ---")
question = "Jackson is clearing out his email inbox after noticing he has a whole lot of emails that he no longer needs and can delete. While he is cleaning his inbox, though, he keeps getting more emails. While he deletes 50 emails he gets another 15 sent to him. While he deletes 20 more he receives 5 more emails. After he is done deleting all his old emails he has just the new emails left, including 10 more that were sent to him. How many emails are there in Jackson's inbox now?"
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant. Final answer format: {"answer": <number>}.'},
    {'role': 'user', 'content': f'{SPECIAL_TOKENS["question"]} {question}'}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt += SPECIAL_TOKENS["solver"]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=False,
    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eod"])]
)

print(tokenizer.decode(out[0], skip_special_tokens=False))