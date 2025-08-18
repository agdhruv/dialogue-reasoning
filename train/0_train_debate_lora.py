from pathlib import Path
import torch
import random
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from datetime import datetime

# Data and model paths
DATA_DIR = Path("../datagen/generated_data")
OUTDIR = Path("lora_debate_ckpts")
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Tokens added in the training data
NEW_TOKENS = ["<|question|>", "<|solver|>", "<|critic|>", "<|endofturn|>", "<|endofdialogue|>"]

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
added = tokenizer.add_tokens(NEW_TOKENS, special_tokens=False)
if added != len(NEW_TOKENS):
    print(f"Warning: some NEW_TOKENS already existed in the tokenizer")
tokenizer.pad_token = tokenizer.eos_token # We need a pad token

# Dataset
raw = load_dataset("json", data_files={
    "train": str(DATA_DIR / "train.txt"),
    "val": str(DATA_DIR / "val.txt")
})

MAX_LEN = 2048

def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        max_length=MAX_LEN,
        truncation=True,
        padding=False # let DataCollator handle padding per-batch
    )

tokenized = raw.map(tokenize_fn, batched=True, num_proc=os.cpu_count(), remove_columns=['text'])

# Data collator (causal LM)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 4-bit quantization
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    device_map="auto"
)

# resize for new tokens
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA
lora_cfg = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    modules_to_save=["embed_tokens", "lm_head"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# Training arguments
args = TrainingArguments(
    output_dir=str(OUTDIR),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,        # effective batch 32
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
    logging_steps=1,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name=f"debate-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    seed=SEED
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=3,        # #evals with no improvement
    early_stopping_threshold=0.0      # minimum loss delta (0 = any improvement)
)


# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=collator,
    callbacks=[early_stop]
)

# Train
trainer.train()
trainer.save_model(OUTDIR / "final")
tokenizer.save_pretrained(OUTDIR / "final")

print("\n✔ training finished\n")