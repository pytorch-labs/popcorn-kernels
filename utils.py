"""
Util file
"""
import re

# TODO
# query a LLM
# Extract Code

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
    


    