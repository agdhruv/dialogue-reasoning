import json
import re
from pathlib import Path
from tqdm import tqdm
from openai import AzureOpenAI

EXTRACTION_MODEL = "gpt-4o_2024-11-20"

def extract_answer_using_llm(client: AzureOpenAI, method: str, response_data: dict | list) -> str | None:
    """
    Uses OpenAI's JSON mode to extract a numerical answer from a text response.

    Args:
        client: An instance of AzureOpenAI.
        method: The method name ("zero_shot", "cot", or "debate").
        response_data: The full response data to parse the response or conversation.

    Returns:
        The extracted numerical answer as a string, or None if no answer is found.
    """
    system_prompt = (
        "You are an expert at extracting information. From the user's text, "
        "find the final numerical answer. Respond with a JSON object like this: {\"answer\": <number>}. "
        "The value should be the extracted number (as a number, not a string). If no numerical answer is found, the value should be null."
    )

    text_to_extract_from = get_response_text(method, response_data)

    if not text_to_extract_from:
        return None
    
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_to_extract_from}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        response_content = response.choices[0].message.content
        data = json.loads(response_content)
        answer = data.get("answer")
        
        # The answer might be a number, convert to string for consistency
        return str(answer) if answer is not None else None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error during answer extraction: {e}")
        # As a fallback, you could implement a regex search here
        return None

def extract_answer_with_regex(method_name: str, response_data: dict | list) -> str | None | list:
    """
    Uses regex to extract numerical answers by searching for {"answer": number} pattern.
    
    Args:
        method_name: The method name ("zero_shot", "cot", or "debate").
        response_data: For debate conversations, the full response data to parse turns.
        
    Returns:
        For zero_shot/cot: The extracted numerical answer as a string, or None if no answer is found.
        For debate: List of tuples [(answer, turn_id), ...] where turn_id is 0-indexed.
    """
    # For debate conversations, extract from each Solver turn
    if method_name == "debate":
        if not response_data:
            return []
        
        answers = []
        pattern = r'\{"answer":\s*([0-9]+(?:\.[0-9]+)?)\}'
        
        for turn_id, message in enumerate(response_data):
            if message.get("speaker") == "Solver":
                content = message.get("content", "")
                match = re.findall(pattern, content)
                if match:
                    answers.append((match[-1], turn_id))
        
        return answers
    
    # For zero_shot and cot, search in the single text
    text_to_extract_from = get_response_text(method_name, response_data)
    if not text_to_extract_from:
        return None
    
    # For zero_shot and cot, search in the single text
    pattern = r'\{"answer":\s*([0-9]+(?:\.[0-9]+)?)\}'
    match = re.search(pattern, text_to_extract_from)
    
    if match:
        return match.group(1)
    
    return None

def get_response_text(method_name: str, response_data: dict | list) -> str:
    """Extracts the relevant text content from the saved response data."""
    if method_name in ["zero_shot", "cot"]:
        return response_data['choices'][0]['message']['content']
    elif method_name == "debate":
        conversation_string = ""
        for message in response_data:
            conversation_string += f"{message['speaker']}: {message['content']}\n"
        return conversation_string

def extract_results(client: AzureOpenAI, input_dir: Path, output_dir: Path, extraction_method: str):
    """
    Reads result files, extracts answers, and saves them to a new directory.
    """
    methods = [dir.name for dir in input_dir.iterdir() if dir.is_dir()][::-1]
    
    for method in methods:
        method_input_dir = input_dir / method
        method_output_dir = output_dir / method
        method_output_dir.mkdir(parents=True, exist_ok=True)
        
        result_files = list(method_input_dir.glob("*.json"))
        for file_path in tqdm(result_files, desc=f"Extracting {method} answers"):
            index = int(file_path.stem)
            output_file = method_output_dir / f"{index}.json"

            if output_file.exists():
                continue

            with open(file_path, "r") as f:
                response_data = json.load(f)

            if extraction_method == "regex":
                extracted_answer = extract_answer_with_regex(method, response_data)
            elif extraction_method == "llm":
                extracted_answer = extract_answer_using_llm(client, method, response_data)
            else:
                raise ValueError(f"Invalid extraction method: {extraction_method}")

            final_data = {
                "question_index": index,
                "extracted_answer": extracted_answer
            }
            
            with open(output_file, "w") as f:
                json.dump(final_data, f, indent=4)