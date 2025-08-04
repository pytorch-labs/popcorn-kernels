# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

def get_model_names_from_folder(folder_path: str) -> list[str]:
    """
    Get a list of model names from a folder by stripping .py extension from filenames.

    Args:
        folder_path (str): Path to the folder containing model files

    Returns:
        list[str]: List of model names without .py extension
    """
    # Get all .py files in the folder
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
    # Strip .py extension and remove leading numbers/underscore
    model_names = [os.path.splitext(f)[0].split('_', 1)[1] for f in model_files]
    
    return model_names

def main():
    # Level 2
    KernelBench_problem_dir = "/sailhome/simonguo/cuda_monkeys/KernelBenchInternal/KernelBench/level2/"
    model_names = get_model_names_from_folder(KernelBench_problem_dir)

    # Save model names to level2.json
    with open('kernelbench_level2_problems.json', 'w') as f:
        json.dump(model_names, f, indent=4)
    print(model_names)

if __name__ == "__main__":
    main()
