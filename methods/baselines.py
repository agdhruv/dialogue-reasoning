from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion

def zero_shot_answer(client: OpenAI | AzureOpenAI, question: str, model_name: str) -> ChatCompletion:
    """
    Generates a direct, zero-shot answer using the provided client.
    """
    system_prompt = "Solve the following problem."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

def cot_answer(client: OpenAI | AzureOpenAI, question: str, model_name: str) -> ChatCompletion:
    """
    Generates a chain-of-thought answer using the provided client.
    """
    system_prompt = "Solve the following problem. Think step-by-step."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response