# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import gc
import glob
import json
import os
import signal
import subprocess
import tempfile
import time
import tokenize
import uuid
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.multiprocessing as mp

from code_transforms import transform_get_functions
from create_triton_data import write_modified_program
from filelock import FileLock  # New import for file locking
from filters import check_non_functional, check_single_return
from run_and_check import evaluate_ref_and_kernel_correctness
from torch.utils._get_clean_triton import get_clean_triton
from tqdm import tqdm
from utils import lint_code_directory, run_ruff_on_code

"""
Triton Code Dataset Generator   

This script processes PyTorch code and Triton kernels to create a dataset of valid
Python-Triton function pairs. It extracts code, cleans it, applies transformations,
and validates the resulting pairs for correctness.
"""


class Timer:
    """Simple timer for measuring execution time of code blocks."""

    def __init__(self, description: str):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.description} took {elapsed_time:.2f} seconds")


# Helper functions for defaultdict - must be at module level for pickling
def default_dict_factory() -> DefaultDict:
    """Create a new defaultdict(list) for nested defaultdicts."""
    return defaultdict(list)


def remove_python_comments(source: str) -> str:
    """
    Remove all comments from Python source code without altering other formatting.

    Args:
        source: The Python source code as a string

    Returns:
        The source code with all comments removed
    """
    source_bytes = source.encode("utf-8")
    stream = BytesIO(source_bytes)
    tokens = tokenize.tokenize(stream.readline)

    result = []
    last_lineno, last_col = 1, 0

    for token in tokens:
        token_type = token.type
        token_string = token.string
        start_line, start_col = token.start
        end_line, end_col = token.end

        # Skip encoding and endmarker tokens
        if token_type in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue

        if token_type == tokenize.COMMENT:
            # Skip comments
            last_lineno, last_col = end_line, end_col
            continue

        # Handle spacing between tokens
        if start_line > last_lineno:
            last_col = 0

        if start_col > last_col:
            result.append(" " * (start_col - last_col))

        # Add the token
        result.append(token_string)
        last_lineno, last_col = end_line, end_col

    return "".join(result)


# No longer needed since we'll directly create file tasks in extract_code_links


def process_file(
    file_data: Dict,
    cleaned_triton_dir: str,
    tests_dir: str,
    synthetic_data_dir: str,
    lock_dir: str,
) -> Optional[Dict]:
    """
    Process a single code file.

    Args:
        file_data: Dictionary with file information
        cleaned_triton_dir: Directory for cleaned triton code
        tests_dir: Directory for test files
        synthetic_data_dir: Directory for synthetic data
        lock_dir: Directory for lock files

    Returns:
        Tracking dictionary or None if processing failed
    """
    repo_name = file_data["repo_name"]
    module_name = file_data["module_name"]
    code_file = file_data["code_file"]
    file_name = f"{repo_name}.{module_name}.py"
    triton_file_path = os.path.join(cleaned_triton_dir, file_name)

    # Create a unique lock file for this module
    lock_file_path = os.path.join(lock_dir, f"{repo_name}.{module_name}.lock")

    is_synthetic = False
    if repo_name == "POPCORN_SYNTHETIC_DATA":
        test_file_path = os.path.join(synthetic_data_dir, file_name)
        is_synthetic = True
    else:
        test_file_path = os.path.join(tests_dir, file_name)

    # Use file lock to prevent race conditions when checking/creating files
    with FileLock(lock_file_path):
        if os.path.exists(triton_file_path) and os.path.exists(test_file_path):
            return {
                "repo_name": repo_name,
                "module_name": module_name,
                "code_file": code_file,
                "python_file": test_file_path,
                "synthetic": is_synthetic,
            }
        else:
            try:
                print(f"running {code_file}")
                # Run the code file to generate triton code
                env = os.environ.copy()
                env["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"
                env["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
                num_gpus = torch.cuda.device_count()
                process_id = os.getpid()
                gpu_id = process_id % num_gpus

                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                subprocess.run(["python", code_file], env=env, check=True)

                # Get clean triton safely with lock
                temp_output = f"{triton_file_path}.{uuid.uuid4().hex}.tmp"
                get_clean_triton(code_file, temp_output)
                os.replace(temp_output, triton_file_path)

                return {
                    "repo_name": repo_name,
                    "module_name": module_name,
                    "code_file": code_file,
                    "python_file": test_file_path,
                    "synthetic": is_synthetic,
                }

                # Verify files exist
                assert os.path.exists(code_file)
                assert os.path.exists(f"{triton_file_path}")
            except Exception as e:
                print(f"Failed to clean triton code for {file_name} with error : {e}")
                return None


def prepare_linted_code(
    tracking_dict: Dict,
    cleaned_triton_dir: str,
    linted_triton_dir: str,
    lock_dir: str,  # New parameter for lock directory
) -> Dict:
    """
    Prepare linted version of triton code.

    Args:
        tracking_dict: Dictionary with module information
        cleaned_triton_dir: Directory for cleaned triton code
        linted_triton_dir: Directory for linted triton code
        lock_dir: Directory for lock files

    Returns:
        Updated tracking dictionary
    """
    repo_name = tracking_dict["repo_name"]
    module_name = tracking_dict["module_name"]
    file_name = f"{repo_name}.{module_name}.py"
    cleaned_file = os.path.join(cleaned_triton_dir, file_name)
    linted_file = os.path.join(linted_triton_dir, file_name)

    # Create a unique lock file for this module's linting operation
    lock_file_path = os.path.join(lock_dir, f"{repo_name}.{module_name}.lint.lock")

    # Use atomic file operations with locking
    with FileLock(lock_file_path):
        if os.path.exists(cleaned_file):
            with open(cleaned_file, "r") as f:
                code = f.read()

            # Remove comments
            try:
                code = remove_python_comments(code)
            except Exception as e:
                print(f"Failed to remove comments for {file_name}: {e}")
                return None

            # Write to a temporary file first for atomicity
            temp_file = f"{linted_file}.{uuid.uuid4().hex}.tmp"
            with open(temp_file, "w") as f:
                f.write(code)

            with open(linted_file, "w") as f:
                f.write(code)
                # Atomic rename
                os.replace(temp_file, linted_file)
    tracking_dict["linted_code_file"] = linted_file
    return tracking_dict


def process_dataset_item(
    tracking_dict: Dict,
) -> Optional[Dict]:
    """
    Process a dataset item to create Python-Triton code pair.

    Args:
        tracking_dict: Dictionary with module information
        lock_dir: Directory for lock files

    Returns:
        Dataset entry or None if processing failed
    """
    module_file = tracking_dict["python_file"]
    triton_file = tracking_dict["linted_code_file"]
    original_triton_code_file = tracking_dict["code_file"]
    entry_point = tracking_dict["module_name"]

    try:
        with open(original_triton_code_file, "r") as f:
            original_triton_code = f.read()

        with open(module_file, "r") as f:
            python_code = f.read()

        with open(triton_file, "r") as f:
            triton_code = f.read()

        # Patch and lint the triton code
        patched_triton_code = write_modified_program(
            python_code, entry_point, triton_code
        )
        patched_triton_code = run_ruff_on_code(patched_triton_code)

        dataset_entry = {
            "entry_point": entry_point,
            "original_triton_code": original_triton_code,
            "python_code": python_code,
            "triton_code": patched_triton_code,
            "repo_name": tracking_dict["repo_name"],
            "module_name": tracking_dict["module_name"],
            "synthetic": tracking_dict["synthetic"],
        }
        return dataset_entry
    except Exception as e:
        print(f"Failed to patch triton code for {module_file}: {e}")
        print(
            f"debug by running \n python create_triton_data.py --pytorch-program {module_file} --triton-program {triton_file} --entry-point {entry_point}\n"
        )
        return None


def check_dataset_entry(
    dataset_entry: Dict,
    lock_dir: str = None,  # New optional parameter for lock directory
) -> Tuple[Dict, bool]:
    """
    Check if a dataset entry is valid by evaluating correctness.

    Args:
        dataset_entry: Dictionary with code information
        lock_dir: Directory for lock files

    Returns:
        Tuple of (dataset_entry, is_valid)
    """
    test = False

    # Basic validation
    required_keys = [
        "entry_point",
        "original_triton_code",
        "python_code",
        "triton_code",
    ]
    if not all(key in dataset_entry for key in required_keys):
        return dataset_entry, test

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"eval_{dataset_entry['entry_point']}_"
        ) as temp_dir:
            # Write files to temp directory
            py_file = os.path.join(temp_dir, "ref.py")
            triton_file = os.path.join(temp_dir, "kernel.py")

            with open(py_file, "w") as f:
                f.write(dataset_entry["python_code"])

            with open(triton_file, "w") as f:
                f.write(dataset_entry["triton_code"])

            # Evaluate correctness with isolated files
            test = evaluate_ref_and_kernel_correctness(
                dataset_entry["python_code"],
                dataset_entry["triton_code"],
                dataset_entry["entry_point"],
                num_trials=10,
            )
    except Exception as e:
        test = False
        print(f"Failed to evaluate {dataset_entry['entry_point']}: {e}")
    finally:
        return dataset_entry, test


def apply_check_single_return(
    dataset_entry: Dict,
) -> Tuple[bool, Dict]:
    """
    Check if code has a single return statement.

    Args:
        dataset_entry: Dictionary with code information

    Returns:
        Tuple of (is_single_return, dataset_entry)
    """
    # Create a temporary directory for isolation
    temp_dir = tempfile.mkdtemp(prefix=f"check_{dataset_entry['entry_point']}_")
    try:
        # Write files to temp directory
        py_file = os.path.join(temp_dir, "ref.py")
        triton_file = os.path.join(temp_dir, "kernel.py")

        with open(py_file, "w") as f:
            f.write(dataset_entry["python_code"])

        with open(triton_file, "w") as f:
            f.write(dataset_entry["triton_code"])

        # Check functionality with isolated files
        check = check_single_return(
            dataset_entry["python_code"],
            dataset_entry["entry_point"],
        )
    finally:
        # Clean up temp directory
        import shutil

        shutil.rmtree(temp_dir)

    return check, dataset_entry


def apply_check_non_functional(
    dataset_entry: Dict,
) -> Tuple[bool, Dict]:
    """
    Check if code is non-functional.

    Args:
        dataset_entry: Dictionary with code information
        lock_dir: Directory for lock files

    Returns:
        Tuple of (is_functional, dataset_entry)
        # in this case functional means that the produced triton code only uses arguments, weights, and biases as inputs
    """
    # Create a temporary directory for isolation
    temp_dir = tempfile.mkdtemp(prefix=f"check_{dataset_entry['entry_point']}_")
    try:
        # Write files to temp directory
        py_file = os.path.join(temp_dir, "ref.py")
        triton_file = os.path.join(temp_dir, "kernel.py")

        with open(py_file, "w") as f:
            f.write(dataset_entry["python_code"])

        with open(triton_file, "w") as f:
            f.write(dataset_entry["triton_code"])

        # Check functionality with isolated files
        check = check_non_functional(
            dataset_entry["python_code"],
            dataset_entry["triton_code"],
            dataset_entry["entry_point"],
        )
    finally:
        # Clean up temp directory
        import shutil

        shutil.rmtree(temp_dir)

    return check, dataset_entry


# Define module-level function for multiprocessing
def _process_and_queue_item(args):
    """Process an item and put results in the queue.

    Args:
        args: Tuple of (item, process_func, result_queue)
    """
    item, process_func, result_queue = args
    result = process_func(item)
    if result is not None:
        if isinstance(result, list):
            for r in result:
                result_queue.put(r)
        else:
            result_queue.put(result)
    return None


def process_items_parallel(
    items: List[Any], process_func: callable, num_workers: int, desc: str = "Processing"
) -> List[Any]:
    """
    Process items in parallel using multiprocessing.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        num_workers: Number of worker processes to use
        desc: Description for progress bar

    Returns:
        List of processed items
    """
    # Use a manager for synchronized collection of results
    manager = mp.Manager()
    result_queue = manager.Queue()

    # Prepare arguments for the worker function
    process_args = [(item, process_func, result_queue) for item in items]

    with Timer(f"Parallel processing - {desc}"):
        with mp.Pool(num_workers) as pool:
            # Process items but don't collect results directly
            list(
                tqdm(
                    pool.imap_unordered(_process_and_queue_item, process_args),
                    total=len(items),
                    desc=desc,
                )
            )
        pool.close()
        pool.join()

    # Collect results from queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return results


def _process_and_filter_item(args):
    """Process an item, filter it, and put in appropriate queue.
    Args:
        args: Tuple of (item_id, item, process_func, passed_queue, failed_queue,
              filter_key, error_queue)
    """
    item_id, item, process_func, passed_queue, failed_queue, filter_key, error_queue = (
        args
    )
    try:
        num_gpus = torch.cuda.device_count()
        gpu_id = item_id % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        result = process_func(item)
        result = process_func(item)
        if result[filter_key]:
            passed_queue.put((item_id, result[1] if filter_key == 0 else result[0]))
        else:
            failed_queue.put((item_id, result[1] if filter_key == 0 else result[0]))
        return True
    except Exception as e:
        # Report the error along with the item ID
        error_queue.put((item_id, str(e)))
        return False


def process_and_filter_parallel(
    items: List[Any],
    process_func: callable,
    num_workers: int,
    desc: str = "Processing",
    filter_key: int = 0,
    timeout: int = 30,  # 30 second timeout per item
) -> Tuple[List[Any], List[Any], Dict[int, str]]:
    """
    Process items in parallel and filter based on a condition.
    Args:
        items: List of items to process
        process_func: Function that returns (condition, item)
        num_workers: Number of worker processes to use
        desc: Description for progress bar
        filter_key: Index to check in the result tuple for filtering
        timeout: Maximum time in seconds for processing a single item
    Returns:
        Tuple of (passed_items, failed_items, error_dict)
    """
    # Use a manager for synchronized collection of results
    manager = mp.Manager()
    passed_queue = manager.Queue()
    failed_queue = manager.Queue()
    error_queue = manager.Queue()

    # Prepare arguments for the worker function with item IDs
    process_args = [
        (i, item, process_func, passed_queue, failed_queue, filter_key, error_queue)
        for i, item in enumerate(items)
    ]

    with Timer(f"Parallel processing and filtering - {desc}"):
        completed_tasks = 0
        errors = {}

        with mp.Pool(num_workers, maxtasksperchild=1) as pool:
            results = [
                pool.apply_async(_process_and_filter_item, (arg,))
                for arg in process_args
            ]

            # Monitor results with progress bar
            with tqdm(total=len(items), desc=desc) as pbar:
                for result in results:
                    try:
                        # Wait for result with timeout
                        success = result.get(timeout=timeout)
                        completed_tasks += 1
                        pbar.update(1)
                    except mp.TimeoutError:
                        # Handle timeout case
                        pbar.update(1)
                    except Exception as e:
                        # Handle any other exceptions
                        pbar.update(1)

        # Clean pool resources
        pool.close()
        pool.join()

    # Collect results from queues
    passed_items = []
    while not passed_queue.empty():
        item_id, item = passed_queue.get()
        passed_items.append(item)

    failed_items = []
    while not failed_queue.empty():
        item_id, item = failed_queue.get()
        failed_items.append(item)

    # Collect error information
    error_dict = {}
    while not error_queue.empty():
        item_id, error_msg = error_queue.get()
        error_dict[item_id] = error_msg

    # Report completion status
    print(f"Completed {completed_tasks}/{len(items)} tasks")
    print(f"Errors: {len(error_dict)}")

    return passed_items, failed_items, error_dict


class TritonDatasetGenerator:
    """Main class for generating Triton datasets."""

    def __init__(
        self,
        run_dir: str,
        num_workers: int,
        num_concurrent_gpu_jobs: int,
        limit: Optional[int] = None,
    ):
        self.run_dir = run_dir
        self.num_workers = num_workers
        self.num_concurrent_gpu_jobs = num_concurrent_gpu_jobs
        self.limit = limit

        # Directory structure
        self.tests_dir = os.path.join(run_dir, "cleaned_pytorch_modules")
        self.cache_dir = os.path.join(run_dir, "inductor_cache")
        self.synthetic_data_dir = os.path.join(run_dir, "synthetic_modules")
        self.cleaned_pytorch_modules_dir = os.path.join(
            run_dir, "cleaned_pytorch_modules"
        )
        self.cleaned_triton_dir = os.path.join(run_dir, "cleaned_triton")
        self.linted_triton_dir = os.path.join(run_dir, "linted_triton")
        self.log_path = os.path.join(run_dir, "inductor_logs")
        self.intermediate_datasets_dir = os.path.join(run_dir, "intermediate_datasets")
        self.datasets_dir = os.path.join(run_dir, "datasets")

        # Add lock directory for synchronization
        self.lock_dir = os.path.join(run_dir, "locks")

        # Create necessary directories
        os.makedirs(self.cleaned_pytorch_modules_dir, exist_ok=True)
        os.makedirs(self.cleaned_triton_dir, exist_ok=True)
        os.makedirs(self.linted_triton_dir, exist_ok=True)
        os.makedirs(self.intermediate_datasets_dir, exist_ok=True)
        os.makedirs(self.synthetic_data_dir, exist_ok=True)
        os.makedirs(self.lock_dir, exist_ok=True)  # Create lock directory

        print("Instance of TritonDatasetGenerator created")

    def extract_code_links(self) -> Tuple[List[Dict], List[str]]:
        """
        Extract code file links from log files and create file-level tasks directly.

        Returns:
            Tuple of (file_tasks, code_files)
        """
        print("Extracting code links from logs")
        with Timer("Extracting code links from logs"):
            # Create a flat list of file tasks directly
            file_tasks = []
            unique_code_files = set()  # Track unique files with a set

            # We'll also maintain a mapping for duplicate checking
            module_file_map = {}  # {repo_name.module_name: count}

            for file_path in glob.glob(os.path.join(self.log_path, "*.txt")):
                with open(file_path, "r") as f:
                    for line in f:
                        if f"Output code written to: {self.cache_dir}" in line:
                            # Parse repository and module name from filename
                            name_string = os.path.basename(file_path)
                            name_string = name_string[:-4]  # remove .txt
                            name_string = name_string.split(".")
                            repo_name = name_string[0]
                            module_name = name_string[1]
                            module_key = f"{repo_name}.{module_name}"

                            # Extract code file path
                            code_file = line.split("Output code written to:")[1].strip()

                            # Track modules with multiple files
                            module_file_map.setdefault(module_key, 0)
                            module_file_map[module_key] += 1

                            if code_file not in unique_code_files:
                                if module_file_map[module_key] > 1:
                                    print(
                                        f"Found {module_file_map[module_key]} code files for {module_key}"
                                    )
                                    continue

                                # Create file task directly
                                file_tasks.append(
                                    {
                                        "repo_name": repo_name,
                                        "module_name": module_name,
                                        "code_file": code_file,
                                    }
                                )
                                unique_code_files.add(code_file)

        # Convert set back to list for return value compatibility
        code_files = list(unique_code_files)
        return file_tasks, code_files

    def generate_dataset(self) -> Optional[Tuple[List[Dict], List[Dict], List[Dict]]]:
        """
        Generate the dataset by processing code files.

        Returns:
            The following 3 lists dataset entries, dataset entries for scrape, and dataset entries for synthetic data
        """
        with Timer("Total dataset generation"):
            # Extract code links from logs and get file tasks directly
            file_tasks, code_files = self.extract_code_links()

            print(f"Found {len(file_tasks)} files to process")

            # Apply limit at the file level
            if self.limit:
                print(
                    f"Limiting to {self.limit} files of total {len(file_tasks)} files"
                )
                file_tasks = file_tasks[: self.limit]

            # Save file tasks for reference
            file_tasks_path = os.path.join(
                self.intermediate_datasets_dir, "file_tasks.json"
            )

            # Use lock when writing to JSON file
            with FileLock(os.path.join(self.lock_dir, "file_tasks.lock")):
                with open(file_tasks_path, "w") as f:
                    json.dump(file_tasks, f, indent=4)

            # Process files in parallel
            with Timer("Processing files"):
                process_file_func = partial(
                    process_file,
                    cleaned_triton_dir=self.cleaned_triton_dir,
                    tests_dir=self.tests_dir,
                    synthetic_data_dir=self.synthetic_data_dir,
                    lock_dir=self.lock_dir,
                )
                tracking_dicts = process_items_parallel(
                    file_tasks,
                    process_file_func,
                    self.num_concurrent_gpu_jobs,
                    "Processing files",
                )

            # Filter out None values
            tracking_dicts = [d for d in tracking_dicts if d is not None]

            print(
                f"Found {len(tracking_dicts)} triton modules that were able to be cleaned"
            )

            lint_code_directory(self.cleaned_pytorch_modules_dir)

            # Prepare linted triton code
            with Timer("Preparing linted triton code"):
                prepare_linted_code_func = partial(
                    prepare_linted_code,
                    cleaned_triton_dir=self.cleaned_triton_dir,
                    linted_triton_dir=self.linted_triton_dir,
                    lock_dir=self.lock_dir,  # Pass lock directory
                )
                tracking_dicts = process_items_parallel(
                    tracking_dicts,
                    prepare_linted_code_func,
                    self.num_workers,
                    "Preparing linted code",
                )

            # Filter out None values again
            tracking_dicts = [d for d in tracking_dicts if d is not None]

            # Apply linter to the triton code - use lock
            with FileLock(os.path.join(self.lock_dir, "lint_triton.lock")):
                lint_code_directory(self.linted_triton_dir)

            # Process dataset items
            with Timer("Processing dataset items"):
                process_dataset_item_func = partial(
                    process_dataset_item,
                )
                raw_dataset = process_items_parallel(
                    tracking_dicts,
                    process_dataset_item_func,
                    self.num_workers,
                    "Creating dataset",
                )

            # Filter out None values
            raw_dataset = [d for d in raw_dataset if d is not None]

            print(f"Dataset before filtering: {len(raw_dataset)}")

            with open(
                os.path.join(self.intermediate_datasets_dir, "raw_dataset.json"),
                "w",
            ) as f:
                json.dump(raw_dataset, f, indent=4)

            # Filter out non-functional code
            with Timer("Filtering out non-functional code"):
                apply_check_func = partial(
                    apply_check_non_functional,
                )
                filtered_dataset_non_functional, nonfunctional_data, error_dict = (
                    process_and_filter_parallel(
                        raw_dataset,
                        apply_check_func,
                        self.num_workers,
                        "Filtering out non-functional code",
                    )
                )

            # Filter out pytorch code that has more than one return statement
            with Timer("Filtering outcode with more than one return statement"):
                apply_check_func = partial(
                    apply_check_single_return,
                )
                filtered_dataset_returns, bad_return_data, error_dict = (
                    process_and_filter_parallel(
                        filtered_dataset_non_functional,
                        apply_check_func,
                        self.num_workers,
                        "Filtering with more than one return statement",
                    )
                )

            filtered_dataset = filtered_dataset_returns
            print(f"Dataset after filtering: {len(filtered_dataset)}")

            # Save filtered datasets - use locks
            with Timer("Saving filtered datasets"):
                with open(
                    os.path.join(
                        self.intermediate_datasets_dir,
                        "FILTERED_has_nonfunctional_data.json",
                    ),
                    "w",
                ) as f:
                    json.dump(nonfunctional_data, f, indent=4)

                with open(
                    os.path.join(
                        self.intermediate_datasets_dir,
                        "FILTERED_has_more_than_one_return_data.json",
                    ),
                    "w",
                ) as f:
                    json.dump(bad_return_data, f, indent=4)

                with open(
                    os.path.join(
                        self.intermediate_datasets_dir, "filtered_dataset.json"
                    ),
                    "w",
                ) as f:
                    json.dump(filtered_dataset, f, indent=4)
        # get filtered dataset
        with open(
            os.path.join(self.intermediate_datasets_dir, "filtered_dataset.json"),
            "r",
        ) as f:
            filtered_dataset = json.load(f)
        # Transform code to comply with kernel bench format
        with Timer("Transforming code to kernel bench format"):
            for dataset_entry in tqdm(filtered_dataset, desc="Transforming code"):
                dataset_entry["python_code"] = transform_get_functions(
                    dataset_entry["python_code"]
                )

        # Filter for correctness
        with Timer("Evaluating correctness"):
            dataset, modules_which_failed_correctness, error_dict = (
                process_and_filter_parallel(
                    filtered_dataset,
                    check_dataset_entry,
                    self.num_concurrent_gpu_jobs,
                    "Evaluating correctness",
                    filter_key=1,
                )
            )

        print(
            f"After filtering for correctness produced dataset of size: {len(dataset)}"
        )
        print(
            f"Failed to evaluate {len(modules_which_failed_correctness)} modules for correctness"
        )

        # Save failed modules - use lock
        with Timer("Saving failed modules"):
            with open(
                os.path.join(
                    self.intermediate_datasets_dir,
                    "failed_for_correctness_modules.json",
                ),
                "w",
            ) as f:
                json.dump(modules_which_failed_correctness, f, indent=4)

        dataset_scrape = copy.deepcopy(
            [entry for entry in dataset if entry["synthetic"] is False]
        )
        dataset_synthetic = copy.deepcopy(
            [entry for entry in dataset if entry["synthetic"] is True]
        )
        # Add UUID to each entry
        for i, dataset_entry in enumerate(dataset):
            dataset_entry["uuid"] = i
        for i, dataset_entry in enumerate(dataset_scrape):
            dataset_entry["uuid"] = i
        for i, dataset_synthetic_entry in enumerate(dataset_synthetic):
            dataset_synthetic_entry["uuid"] = i
        print(f"Scrape dataset size: {len(dataset_scrape)}")
        print(f"Synthetic dataset size: {len(dataset_synthetic)}")
        print(f"Final dataset size: {len(dataset)}")
        return dataset, dataset_scrape, dataset_synthetic


def main():
    """Main function to parse arguments and run the dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a dataset of Python-Triton code pairs"
    )
    parser.add_argument(
        "--run-dir",
        default="./runs/run1",
        help="Directory for run artifacts and outputs",
    )
    # number of CPU jobs to run
    parser.add_argument(
        "--jobs", type=int, default=8, help="Number of parallel workers / jobs"
    )
    # number of concurrent GPU jobs to run, this might be necessary as we might run out of GPU memory
    parser.add_argument(
        "--num-concurrent-gpu-jobs",
        type=int,
        default=8,
        help="Number of concurrent GPU jobs to run",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of repos to process"
    )
    parser.add_argument(
        "--output-file", type=str, default="dataset.parquet", help="Output file name"
    )
    args = parser.parse_args()

    datasets_dir = os.path.join(args.run_dir, "datasets")
    lock_dir = os.path.join(args.run_dir, "locks")

    # Create output directories
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(lock_dir, exist_ok=True)
    output_file = os.path.join(datasets_dir, args.output_file)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    mp.set_start_method("spawn", force=True)

    # Generate the dataset
    with Timer("Overall execution"):
        generator = TritonDatasetGenerator(
            args.run_dir, args.jobs, args.num_concurrent_gpu_jobs, args.limit
        )
        dataset, dataset_scrape, dataset_synthetic = generator.generate_dataset()

        # Save the dataset as a parquet file - use locks
        with Timer("Saving dataset to parquet"):
            if len(dataset_scrape) > 0:
                df_scrape = pd.DataFrame(dataset_scrape)
                output_file_scrape = os.path.join(
                    datasets_dir, "scrape_dataset.parquet"
                )
                df_scrape.to_parquet(output_file_scrape, index=True)

            if len(dataset_synthetic) > 0:
                df_synthetic = pd.DataFrame(dataset_synthetic)
                output_file_synthetic = os.path.join(
                    datasets_dir, "synthetic_dataset.parquet"
                )
                df_synthetic.to_parquet(output_file_synthetic, index=True)

            if len(dataset) > 0:
                df = pd.DataFrame(dataset)
                df.to_parquet(output_file, index=True)


if __name__ == "__main__":
    main()
