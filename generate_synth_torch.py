"""
Demonstrations of how to genreate torch models synthetically


Currently just does one example
"""

# import operators that we defined
from operators import core_operators, compound_operators, supporting_operators
import subprocess
import re
import random
import os

from utils import extract_last_code

import pydra

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
        self.program_dir = "./synth_torch_generations"

        self.verbose = False


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
    print(f"Pattern: {pattern}")
    
    # Step 2. Query the LLM and generate a synthetic program

    # Step 3. Make sure this program is valid
    # Run the torch module as well as if could be torch.compile
    



    # TODO: check this is not in KernelBench (test set)
    # Step 4. Save the program

    return True



@pydra.main(SynthConfig)
def main(config: SynthConfig):
    generate_synth_torch_single(config)


if __name__ == "__main__":
    main()