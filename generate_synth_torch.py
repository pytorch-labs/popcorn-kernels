"""
Generate torch models synthetically

"""

# import operators that we defined
import subprocess
import re
import traceback

import random
import os
import dotenv
import tomli
import pydra
import shutil
import json
import multiprocessing as mp
import copy
from functools import partial
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
from operators import core_operators, compound_operators, supporting_operators
from tqdm import tqdm
from utils import extract_final_pattern, extract_last_code, generate_gemini, generate_local_server_openai, maybe_multiprocess, test_synthetic_model, maybe_multithread, num_generations_in_dir
from typing import Tuple
import threading
from pathlib import Path
import numpy as np
from multiprocessing import Value

# Create a file lock for thread-safe logging
log_lock = threading.Lock()

# Replace the global variables with shared counters
num_generated = mp.Value('i', 0)  
num_success = mp.Value('i', 0)

class SynthConfig(pydra.Config):
    def __init__(self):
        super().__init__()


        # Section: Generation Config
        # range of number of core operators
        # self.num_core_ops_range = [0,4]
        self.num_core_ops_range = [1,5]

        # range of number of compound operators
        # self.num_compound_ops_range = [0,5]
        self.num_compound_ops_range = [0,5]

        # range of number of supporting operators
        self.num_supporting_ops_range = [1,8]

        # knob to control the number of operators generated
        self.p_value = 0.3

        # Section: Model Config
        self.model_name = "gemini-2.0-flash"
        self.server_address = "matx2.stanford.edu"
        self.port = 10210 

        # Section: File Config
        # directory to save the generations to
        self.program_dir = "synth_torch_generations"
        self.kernelbench_level2_problem_path = "kernelbench_level2_problems.json"

        # Section: Generation Config
        self.verbose = False
        self.write_to_file = False
        
        self.mode = None

        
    def single_debug(self):
        self.mode = "single"
        self.num_total_samples = 1
        self.num_worker = 1
        self.debug_dir = "synth_torch_debug"
        self.write_to_file = True
        self.verbose = True
        self.program_dir = "synth_torch_generations"

    def parallel(self):
        self.mode = "parallel"
        self.num_total_samples = 100
        self.num_worker = 50
        self.write_to_file = False
        self.program_dir = "/matx/u/simonguo/synth_torch_generations_local_qwen"

    def __repr__(self):
        return f"SynthConfig({self.to_dict()})"


def generate_patterns_pattern(
    operator_lists_with_ranges: list[tuple[list[str], tuple[int, int]]], 
    p_value: float = 0.3
) -> list:
    """
    Generate a pattern by randomly selecting operators from multiple lists.

    operator_lists_with_ranges: A list of tuples, where each tuple contains:
        - A list of operators to choose from
        - A range [min, max] specifying how many operators to select
    Returns:
        A string with the selected operators joined by underscores
    """
    def weighted_random(min_val, max_val, p_value):
        # p = 0.3  # Probability parameter (higher = more skewed to smaller values)
        # if it is higher, than you generate less operators
        # if it is lower, than you generate more operators; more errors, but more unique
        # start high than start it lower
        # Generate a geometric random variable
        steps = 0
        while random.random() > p_value and min_val + steps < max_val:
            steps += 1
        return min_val + steps

    pattern = []
    
    for operators, count_range in operator_lists_with_ranges:
        # Determine how many operators to select from this list
        num_to_select = weighted_random(count_range[0], count_range[1], p_value)
        # num_to_select = random.randint(count_range[0], count_range[1])
        
        # Select random operators from the list
        for _ in range(num_to_select):
            if operators:  # Check if the list is not empty
                op = random.choice(operators)
                pattern.append(op)
    
    return pattern

def check_if_model_exists_in_kernelbench(final_pattern: list[str], kernelbench_problem_set: set[str]) -> bool:
    """
    Check if a model with the given name exists in the directory
    """

    # Convert final_pattern list to a set for membership testing
    
    # Join pattern items with underscore
    joined_pattern_name = "_".join(final_pattern)

    # print(f"Comparing {joined_pattern_name} with KernelBench Level 2 Problem Set")

    # Check if any operation in final_pattern exists in kernelbench_problem_set
    if joined_pattern_name in kernelbench_problem_set:
        print(f"Model {joined_pattern_name} already exists in KernelBench Level 2!... we cannot use this since this will containmiate the data")
        return True
    else:
        return False

def generate_synth_torch_single(
    work_id: int,
    config: SynthConfig,
) -> Tuple[bool, str]:
    """
    Generate a single torch model with synthetically
    """
    
    if config.verbose:
        print(f"Generating Synth Model {work_id} for total: {config.num_total_samples}")
    
    # Step 1. Choose random operators from the lists
    operator_lists_with_ranges = [
        (core_operators, config.num_core_ops_range),
        # (compound_operators, config.num_compound_ops_range),
        (supporting_operators, config.num_supporting_ops_range),
    ]
    
    pattern = generate_patterns_pattern(operator_lists_with_ranges, config.p_value)
    
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

    if config.verbose:
        print(f"Prompting Model to Generate Program with Pattern: {pattern}")
        print(prompt)
    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

    if config.model_name == "local":
        response = generate_local_server_openai(prompt=prompt, 
                                                server_address=config.server_address,
                                                port=config.port, 
                                                model=config.model_name, 
                                                temperature=0.7, 
                                                max_tokens=2048,
                                                verbose=config.verbose
                                                )
    else: # check openai or gemini type
        response = generate_gemini(prompt=prompt, model=config.model_name, verbose=config.verbose)

    if config.verbose:
        print(response)

    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, "response.txt"), "w") as f:
            f.write(response)

    # Update the counters using with blocks for thread safety
    if config.mode == "parallel":
        with num_generated.get_lock():
            num_generated.value += 1

    code = extract_last_code(response, "python")
    final_pattern = extract_final_pattern(response)

    if config.verbose:
        print("Code Generation Success")
        print(f"Final Pattern (Ordered): {final_pattern}")
    if not (code and final_pattern):
        if config.verbose:
            print("No code or final pattern in response")
        return False, "extraction_failure"

    # Check this is not in KernelBench (test set)
    kernelbench_problem_set = set(json.load(open(config.kernelbench_level2_problem_path)))
    if check_if_model_exists_in_kernelbench(final_pattern, kernelbench_problem_set):
        return False

    # Step 3. Make sure this program is valid

    proposed_entry_point = f"SynthModel_{'_'.join(final_pattern)}"

    if os.path.exists(os.path.join(REPO_DIR, config.program_dir, f"{proposed_entry_point}.py")):
        if config.verbose:
            print(f"Model {proposed_entry_point} already exists in {config.program_dir}!")
                
        # Add numeric suffix to handle duplicates
        suffix = 1
        while os.path.exists(os.path.join(REPO_DIR, config.program_dir, f"{proposed_entry_point}_{suffix}.py")):
            suffix += 1
        proposed_entry_point = f"{proposed_entry_point}_{suffix}"
        print(f"Already exists, renaming to {proposed_entry_point}")
        reason = "duplicate"
    

    # according to Sahan's pipeline, these two should be the same
    # file_name = f"SynthModel_{'_'.join(final_pattern)}.py"
    # entry_point = f"SynthModel_{'_'.join(final_pattern)}"

    file_name = f"{proposed_entry_point}.py"
    entry_point = proposed_entry_point

    # Step 4. Swap the forward call with entry point name
    code = code.replace("Model", f"{entry_point}")

    if config.write_to_file:
        with open(os.path.join(REPO_DIR, config.debug_dir, file_name), "w") as f:
            f.write(code)

    # Step 5. Test the model
    # Run the torch module as well as if could be torch.compile

    if config.verbose:
        print(f"Testing Model {entry_point} can pass torch Eager and torch.compile")
    success, error = test_synthetic_model(torch_src=code, entry_point=entry_point)

    if not success:
        if config.verbose:
            print(f"Error: {error}")
        
        # Log only the failed operators to a file with thread safety
        log_path = Path("failed_operators.log")
        with log_lock:
            with open(log_path, "a") as log_file:
                log_file.write(f"{', '.join(str(op) for op in pattern)}\n")
        
        return False, "fail_to_run"
        
    # Step 6. We made it here, let's save the model    
    reason = "success"
    if config.mode == "parallel":
        with num_success.get_lock():
            num_success.value += 1
    write_file_path = os.path.join(REPO_DIR, config.program_dir, file_name)
    if os.path.exists(write_file_path):
        if config.verbose:
            print(f"File {file_name} already exists!")
        reason = "success_overwrite"

    print(f"Successfully generate {file_name} with final pattern {final_pattern} | input pattern {pattern}")
    if config.mode == "parallel":
        num_current_files = num_generations_in_dir(os.path.join(REPO_DIR, config.program_dir))
        print(f"[Stats] {num_success.value} successful synthetic programs out of {num_generated.value} generated, yield {(num_success.value/num_generated.value)*100:.2f}%, current total synthetic programs: {num_current_files}")
    with open(write_file_path, "w") as f:
        f.write(code)

    # Yay successful synthetic generation!
    return True, reason


def generate_synth_torch_single_wrapper(
    work_id: int,
    config: SynthConfig,
) -> Tuple[bool, str]:
    try:
        return generate_synth_torch_single(work_id, config)
    except Exception as e:
        print(f"Error: Issue with generating synth model {work_id} with config {config}")
        print(f"Exception details: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        
        return False, str(e)

@pydra.main(SynthConfig)
def main(config: SynthConfig):
    print(config)

    # Check number of existing files in run directory
    program_dir_full_path = os.path.join(REPO_DIR, config.program_dir)
    existing_file_count = num_generations_in_dir(program_dir_full_path)
    print(f"Found {existing_file_count} existing files in {program_dir_full_path}")

    if config.mode == "single":
        print("Running in single debug mode")
        generate_synth_torch_single(work_id=0, config=config)
    elif config.mode == "parallel":
        print(f"Running in parallel mode, generating {config.num_total_samples} models with {config.num_worker} workers")

        work_ids = list(range(config.num_total_samples))

        # synth_generation_results = maybe_multithread(
        synth_generation_results = maybe_multiprocess(
            func=generate_synth_torch_single_wrapper, 
            instances=work_ids, 
            num_workers=config.num_worker, 
            time_interval=0.1, 
            # extra args
            config=config, 
        )

        # print(synth_generation_results)

       
        # Analyze results
        total = len(synth_generation_results)
        successes = sum(1 for success, _ in synth_generation_results if success)
        failures = total - successes

        # Group successes by reason
        success_reasons = {}
        for success, reason in synth_generation_results:
            if success:
                success_reasons[reason] = success_reasons.get(reason, 0) + 1

        # Group errors by type
        error_types = {}
        for success, error in synth_generation_results:
            if not success:
                error_types[error] = error_types.get(error, 0) + 1
        
        # Print summary
        print("\nResults Summary:")
        print(f"Total attempts: {total}")
        print(f"Successful generations: {successes}")
        print(f"Failed generations: {failures}")
        print(f"Yield rate: {(successes/total)*100:.2f}%")
        
        if error_types:
            print("\nError breakdown:")
            for error_type, count in error_types.items():
                print(f"- {error_type}: {count} occurrences")
        
        if successes:
            print("\nSuccess breakdown:")
            for reason, count in success_reasons.items():
                print(f"- {reason}: {count} occurrences")

    # Show final count of files in run directory
    final_file_count = num_generations_in_dir(program_dir_full_path)
    print(f"\nFinal count: {final_file_count} files in {program_dir_full_path}")
    print(f"Generated {final_file_count - existing_file_count} new files")


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()