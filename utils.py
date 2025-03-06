"""
Util file
"""
import re
import ast, astor
import os
from google import genai
from google.genai import types
import torch

from typing import Any, List, Tuple, Dict, Optional

def generate_gemini(prompt: str, model: str = "gemini-2.0-flash"):
    """
    Querying Gemini API
    """
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

    # show a select few attributes
    config_dict = {
        "temperature": generate_content_config.temperature,
        "top_p": generate_content_config.top_p,
        "top_k": generate_content_config.top_k,
        "max_output_tokens": generate_content_config.max_output_tokens,
        "response_mime_type": generate_content_config.response_mime_type
    }
    print(f"Querying Gemini {model} with config: {config_dict}")

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    return response.text

def extract_last_code(output_string: str, code_language_type: str) -> str:
    """
    Extract the last code block from the output string
    """
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

def swap_forward_call(pytorch_code, entry_point, generate_forward_call):
    """
    Sahan's AST based swapping code (did not test this yet)
    """
    # find the forward call and remove it and return the rest of the code
    tree = ast.parse(pytorch_code)

    # Helper function to rename Name nodes
    def rename_references(node):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Name) and child_node.id == entry_point:
                child_node.id = f"{entry_point}New"

    for node in ast.walk(tree):
        # find the module definition for the entry point
        if isinstance(node, ast.ClassDef) and node.name == entry_point:
            # Rename the class definition
            node.name = f"{entry_point}New"

            # Scan all nodes in the class body to replace references to entry_point
            for child in node.body:
                rename_references(child)

                # Additionally, replace the forward method
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    node.body.remove(child)
                    # replace with the generate_forward_call function with proper indentation
                    node.body.append(ast.parse(generate_forward_call).body[0])

    # Also scan for references outside the class definition
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == entry_point:
            node.id = f"{entry_point}New"

    return astor.to_source(tree)

def test_synthetic_model(torch_src: str, entry_point: str) -> Tuple[bool, Optional[str]]:
    """
    Tests a synthetic model by loading it with exec, initializing it with get_init_inputs,
    and running it with get_inputs.
    Check if the model can pass torch Eager and torch.compile
    Returns a tuple of (success, error_message)
    """
    try:
        # Read and execute the file content
        namespace = {}
        exec(torch_src, namespace)
        
        
        # Find the model class by name
        model_class = namespace.get(entry_point)
        if not model_class or not issubclass(model_class, torch.nn.Module):
            return False, f"Could not find valid PyTorch model class named {entry_point}"
        
        # Get initialization parameters
        get_init_inputs = namespace.get('get_init_inputs')
        if not get_init_inputs:
            return False, "get_init_inputs function not found in the file"
        
        init_args, init_kwargs = get_init_inputs()
        
        # Initialize the model
        model = model_class(*init_args, **init_kwargs)
        
        # Get input tensors
        get_inputs = namespace.get('get_inputs')
        if not get_inputs:
            return False, "get_inputs function not found in the file"
            
        inputs = get_inputs()

        # Test eager mode
        with torch.no_grad():
            try:
                outputs = model(*inputs)
                
                # Basic validation of outputs
                if not isinstance(outputs, (torch.Tensor, tuple, list)):
                    return False, "Model output is not a tensor or tuple/list of tensors"
            except Exception as e:
                return False, f"Error during eager mode forward pass: {str(e)}"
        
        # Test compiled mode
        try: 
            # default Inductor backend
            compiled_model = torch.compile(model)
            compiled_outputs = compiled_model(*inputs)
            
            # Basic validation of compiled outputs
            if not isinstance(compiled_outputs, (torch.Tensor, tuple, list)):
                return False, "Compiled model output is not a tensor or tuple/list of tensors"
                
            return True, None
        except Exception as e:
            return False, f"Error in compiled mode: {str(e)}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Example file path (adjust according to your project structure)
    file_path = os.path.join(os.getcwd(), "synth_torch_debug/synth_torch_Conv2d_AvgPool2d.py")
    # Ensure file exists before testing
    print(f"Current working directory: {os.getcwd()}")
    assert os.path.exists(file_path), f"Error: File {file_path} does not exist"
    entry_point = "SynthModel_Conv2d_AvgPool2d"
    with open(file_path, 'r') as f:
        file_content = f.read()
    success, error = test_synthetic_model(file_content, entry_point)
    if success:
        print("Model test successful!")
    else:
        print(f"Model test failed: {error}")