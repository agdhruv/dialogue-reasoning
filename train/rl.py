"""RL fine-tuning (RLVR) of the debate LoRA model with GRPO
----------------------------------------------------------------
The script assumes that you have already run `0_train_debate_lora.py` and
have a checkpoint directory under `lora_debate_ckpts/`.  We now run a
second stage of training that optimises the model with the custom *debate*
reward function using Hugging Face TRL's `GRPOTrainer`.

Usage (single-GPU example):
    $ accelerate launch train/rl.py
"""
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
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

def load_finetuned_model():
    """Load base model + LoRA adapters"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    return model


###############################################################################
# Reward function
###############################################################################

def _extract_answer(segment: str) -> int | None:
    """Return the *first* integer answer inside a dialogue segment, else None."""
    answer_regex = re.compile(r'\{"answer":\s*([0-9]+(?:\.[0-9]+)?)\}')
    m = answer_regex.findall(segment)
    if m:
        return int(m[-1])
    return None

def _compute_reward_single(trace: str, gold: int) -> float:
    """
    +1.0  if solver is correct on first try (no critic needed)
    +0.8  if first solver answer is wrong BUT
           (a)  a critic turn appears before the fix AND
           (b)  the next solver answer is correct
    -0.2  otherwise (still wrong at end)
    -0.002 * extra_tokens_after_fix   # length penalty
    """
    # Create a list of segments between solver and critic (just string manipulation)
    # each segment is of the form "<|role|> …"
    s = trace
    segs = s[s.find('<|solver|>', s.find('<|solver|>') + 1):].split('<|endofturn|>')[:-1]

    correct_answer_index = None
    for i, seg in enumerate(segs):
        answer = _extract_answer(seg)
        if answer == str(gold):
            correct_answer_index = i
            break
    correct_answer_found = correct_answer_index is not None
    
    if correct_answer_index == 0 and len(segs) == 1:
        return 1.0
    
    if not correct_answer_found:
        return -0.2
    
    # answer was found but not on first try
    if correct_answer_found:
        # need to make sure that there were alternating turns between solver and critic before the answer was found
        # if there were, give high reward but penalize for number of turns
        # if there were not, give zero reward (to avoid reward hacking by not calling the critic at all)
        roles = ['solver' if seg.startswith('<|solver|>') else 'critic' for seg in segs[:correct_answer_index]]
        if len(roles) % 2 == 0:
            if roles[::2] == ['solver'] * (len(roles) // 2) and roles[1::2] == ['critic'] * (len(roles) // 2):
                tokens_after_fix = sum(len(tokenizer.encode(seg)) for seg in segs[correct_answer_index:])
                return 0.8 - 0.002 * (len(roles) // 2) - 0.002 * tokens_after_fix
        else:
            return 0.0

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

policy = load_finetuned_model()

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
