"""RL fine-tuning (RLVR) of the debate LoRA model with GRPO
----------------------------------------------------------------
The script assumes that you have already run `0_train_debate_lora.py` and
have a checkpoint directory under `lora_debate_ckpts/` (the default path
laid out in the supervised-fine-tuning script).  We now run a second stage
of training that optimises the model with the custom *debate* reward
function using Hugging Face TRL's `GRPOTrainer`.

Usage (single-GPU example):
    $ accelerate launch train/rl.py

Feel free to tweak the GRPOConfig hyper-parameters at the bottom.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

###############################################################################
# Model loading helpers
###############################################################################

ADAPTER_DIR = Path("lora_debate_ckpts") / "checkpoint-600"  # last SFT ckpt
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_finetuned_model():
    """Load base model + LoRA adapters exactly as they were during evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    # From this point `model` already contains the LoRA weights.
    return model, tokenizer


###############################################################################
# Reward function
###############################################################################

answer_regex = re.compile(r"\"answer\"\s*:\s*(-?\d+)")


def _extract_first_answer(segment: str) -> int | None:
    """Return the *first* integer answer inside a dialogue segment, else None."""
    m = answer_regex.search(segment)
    if m:
        return int(m.group(1))
    return None


def _compute_reward_single(trace: str, gold: int) -> float:
    """Implements the reward specification from the user prompt (vector-free)."""
    # Split on every solver turn (we skip the initial <|question|> block)
    segs = trace.split("<|solver|>")[1:]

    earliest: int | None = None
    critic_before_fix = False
    tokens_after_fix = 0

    for seg in segs:
        # Each `seg` is like "<|role|> … <|endofturn|> …"
        # Identify role from the prefix, strip it for body parsing.
        role = "solver" if seg.startswith("<|solver|>") else (
            "critic" if seg.startswith("<|critic|>") else "unknown"
        )
        body = seg.split("<|endofturn|>", 1)[-1]

        if earliest is None:
            found = _extract_first_answer(body)
            if found is not None:
                earliest = found
        if role == "critic" and earliest is None:
            critic_before_fix = True
        if earliest is not None:
            tokens_after_fix += len(body.split())

    correct_now = earliest is not None and int(earliest) == gold

    if correct_now and not critic_before_fix:
        return 1.0
    if correct_now and critic_before_fix:
        return 0.8 - 0.002 * tokens_after_fix
    return -0.2


def debate_reward(
    *,
    prompts: List[str],
    completions: List[str],
    gold: List[int],  # passed automatically from the dataset column
    **_: dict,
) -> List[float]:
    """Vectorised wrapper so that GRPOTrainer can call the reward function.

    The trainer passes lists of *equal length*; we compute rewards element-wise.
    """
    rewards: list[float] = []
    for p, c, g in zip(prompts, completions, gold):
        trace = p + c  # full conversation
        rewards.append(_compute_reward_single(trace, g))
    return rewards


###############################################################################
# Dataset preparation (GSM8K prompts + gold answers)
###############################################################################

# We rely on the HF public GSM8K dataset – adjust to your own data if needed.
gsm8k = load_dataset("gsm8k", "main", split="train")


def parse_gold(ans: str) -> int:
    """Extract the final integer after the solution marker '####'."""
    try:
        return int(ans.split("####")[-1].strip())
    except ValueError:
        # Fallback: grab the last integer in the string
        nums = re.findall(r"-?\d+", ans)
        return int(nums[-1]) if nums else 0


def format_prompt(q: str) -> str:
    return f"<|question|>\n{q}\n<|solver|>"


rl_dataset = gsm8k.map(
    lambda row: {  # noqa: D401 – dict comp clearer here
        "prompt": format_prompt(row["question"]),
        "gold": parse_gold(row["answer"]),
    },
    remove_columns=gsm8k.column_names,
)

###############################################################################
# Build trainer and run
###############################################################################

policy, tokenizer = load_finetuned_model()

# Important for autoregressive tasks: left-padding
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = GRPOConfig(
    output_dir="debate_grpo_ckpts",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    beta=0.1,  # enable KL against a ref model automatically
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    remove_unused_columns=False,
    bf16=True,
)

trainer = GRPOTrainer(
    model=policy,
    reward_funcs=debate_reward,
    args=config,
    train_dataset=rl_dataset,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("debate_grpo_ckpts/final")
    tokenizer.save_pretrained("debate_grpo_ckpts/final")
