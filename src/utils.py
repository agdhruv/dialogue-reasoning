import pandas as pd
from datasets import load_dataset

def load_gsm8k_dataset(split: str, top_k: int = None) -> pd.DataFrame:
    """Load the GSM8K dataset from the Hugging Face dataset library."""
    gsm = load_dataset("gsm8k", "main", split=f"{split}[:{top_k}]" if top_k else split)
    gsm_df = gsm.to_pandas()
    gsm_df['gold_answer'] = gsm_df['answer'].str.split("#### ").str[1].str.strip()
    return gsm_df