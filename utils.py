"""
Util file
"""
import re

# TODO
# query a LLM
# Extract Code

import base64
import os
from google import genai
from google.genai import types


def generate_gemini(prompt: str, model: str = "gemini-2.0-flash-lite"):
    client = genai.Client(  
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    print(f"Querying Gemini {model} with config: {generate_content_config}")

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text

def extract_last_code(output_string: str, code_language_type: str) -> str:
    trimmed = output_string.strip()
    # Extracting all occurrences of content between backticks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    last_match = None
    
    # Find the last match
    for code_match in code_matches:
        last_match = code_match
        
    if last_match:
        # Strip leading and trailing whitespace from the extracted code
        code = last_match.group(1).strip()
        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        if code.startswith(code_language_type):
            code = code[len(code_language_type) :].strip()
        return code
    return None

def extract_final_pattern(output_string: str) -> list[str]:
    pattern_match = re.search(r"Final Pattern: \[(.*?)\]", output_string)
    if not pattern_match:
        return None
    
    final_pattern = [
        op.strip().strip("'") 
        for op in pattern_match.group(1).split(',')
    ]

    return final_pattern

