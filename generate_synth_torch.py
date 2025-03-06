"""
Demonstrations of how to genreate torch models synthetically


Currently just does one example

Notes: things I havent' thought about
- how to name them? (ops, orders, numerical?)
- how to make sure they don't already exist in the generations
"""

# import operators that we defined
import subprocess
import re
import random
import os
import dotenv
import tomli
import pydra
import shutil

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
from operators import core_operators, compound_operators, supporting_operators

from utils import extract_final_pattern, extract_last_code, generate_gemini, test_synthetic_model

class SynthConfig(pydra.Config):
    def __init__(self):
        super().__init__()

        # range of number of core operators
        self.num_core_ops_range = [1,3]

        # range of number of compound operators
        self.num_compound_ops_range = [1,2]

        # range of number of supporting operators
        self.num_supporting_ops_range = [1,5]

        # directory to save the generations to
        self.program_dir = "synth_torch_generations"

        self.verbose = False
        self.debug_dir = "synth_torch_debug"
        self.write_to_file = False

def generate_patterns_pattern(
    operator_lists_with_ranges: list[tuple[list[str], tuple[int, int]]]
) -> str:
    """
    Generate a pattern by randomly selecting operators from multiple lists.

    operator_lists_with_ranges: A list of tuples, where each tuple contains:
        - A list of operators to choose from
        - A range [min, max] specifying how many operators to select
    Returns:
        A string with the selected operators joined by underscores
    """
    pattern = []
    
    for operators, count_range in operator_lists_with_ranges:
        # Determine how many operators to select from this list
        num_to_select = random.randint(count_range[0], count_range[1])
        
        # Select random operators from the list
        for _ in range(num_to_select):
            if operators:  # Check if the list is not empty
                op = random.choice(operators)
                pattern.append(op)
    
    return pattern

def generate_synth_torch_single(
    config: SynthConfig,
) -> bool:
    """
    Generate a single torch model with synthetically
    """

    # Step 1. Choose random operators from the lists
    operator_lists_with_ranges = [
        (core_operators, config.num_core_ops_range),
        # (compound_operators, config.num_compound_ops_range),
        (supporting_operators, config.num_supporting_ops_range),
    ]
    pattern = generate_patterns_pattern(operator_lists_with_ranges)
    
    if config.verbose:
        print(f"Pattern to compose program: {pattern}")
    if config.write_to_file:
        # Clear existing files in debug directory
        if os.path.exists(os.path.join(REPO_DIR, config.debug_dir)):
            shutil.rmtree(os.path.join(REPO_DIR, config.debug_dir))
        os.makedirs(os.path.join(REPO_DIR, config.debug_dir), exist_ok=True)

    # Step 2. Query the LLM and generate a synthetic program
    
    with open("prompts/prompts_simon.toml", "rb") as f:
        data = tomli.load(f)  # or tomllib.load(f)
    
    prompt = data["prompt"].replace("{{pattern}}", str(pattern))

    print(f"Prompting Model to Generate Program with Pattern: {pattern}")
    if config.verbose:
        print(prompt)
    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

    response = generate_gemini(prompt)

    if config.verbose:
        print(response)

    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, "response.txt"), "w") as f:
            f.write(response)

    code = extract_last_code(response, "python")
    final_pattern = extract_final_pattern(response)

    print("Code Generation Success")
    print(f"Final Pattern: {final_pattern}")
    if not (code and final_pattern):
        print("Did not find both code or final pattern in response")
        return False

    # Step 3. Make sure this program is valid

    # according to Sahan's pipeline, these two should be the same
    file_name = f"SynthModel_{'_'.join(final_pattern)}.py"
    entry_point = f"SynthModel_{'_'.join(final_pattern)}"
    


    # Step 4. Swap the forward call with entry point name
    code = code.replace("Model", f"{entry_point}")

    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, file_name), "w") as f:
            f.write(code)

    # Step 5. Test the model
    print(f"Testing Model {entry_point} can pass torch Eager and torch.compile")
    success, error = test_synthetic_model(torch_src=code, entry_point=entry_point)

    if not success:
        print(f"Error: {error}")
        return False

    # Step 6. Save the model    
    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.program_dir, file_name), "w") as f:
            f.write(code)

    # Run the torch module as well as if could be torch.compile
       


    # TODO: check this is not in KernelBench (test set)
    # Step 4. Save the program

    return True



@pydra.main(SynthConfig)
def main(config: SynthConfig):
    generate_synth_torch_single(config)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()