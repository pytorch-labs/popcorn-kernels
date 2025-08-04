# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import os
import random
import shutil

import tempfile

import numpy as np
import pandas as pd
import torch


def set_deterministic(seed=42):
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set environment variables for deterministic operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def import_ModelNew_from_code(code_string, entry_point):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.
    """
    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(code_string)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def import_Model_and_args_from_code(code_string, entry_point):

    context = {}
    compile(code_string, "<string>", "exec")
    exec(code_string, context)
    model = context[entry_point]
    get_inputs = context["get_inputs"]
    get_init_inputs = context["get_init_inputs"]
    return model, get_inputs, get_init_inputs


def evaluate_ref_and_kernel_correctness(
    ref_arch_src: str,
    kernel_src: str,
    entry_point: str,
    num_trials: int,
) -> bool:
    """
    Evaluate the correctness of the reference and kernel code
    """
    set_deterministic()
    model, get_inputs, get_init_inputs = import_Model_and_args_from_code(
        ref_arch_src, entry_point
    )
    model_init_args, model_init_kwargs = get_init_inputs()

    # move all inputs to cuda

    # evaluate the reference code
    ref_model = model(*model_init_args, **model_init_kwargs).cuda()
    triton_model_class, temp_file = import_ModelNew_from_code(
        kernel_src, f"{entry_point}New"
    )
    triton_model = triton_model_class(*model_init_args, **model_init_kwargs).cuda()

    # some triton opereations don't like gradients
    for param in triton_model.parameters():
        param.requires_grad_(False)
    # do it on ref model for consistency
    for param in ref_model.parameters():
        param.requires_grad_(False)

    for _ in range(num_trials):
        model_forward_args = get_inputs()
        model_forward_args = [arg.cuda() for arg in model_forward_args]
        ref_output = ref_model(*model_forward_args)
        triton_output = triton_model(*model_forward_args)

        # delete inputs to save memory
        for arg in model_forward_args:
            del arg
        torch.cuda.empty_cache()

        # fix types to iterable as expected
        if isinstance(ref_output, torch.Tensor):
            ref_output = [ref_output]
        if isinstance(triton_output, torch.Tensor):
            triton_output = [triton_output]
        # assert everything is a tensor

        if len(ref_output) != len(triton_output):
            # close file
            temp_file.close()
            return False
        for ref_out, triton_out in zip(ref_output, triton_output):
            if not isinstance(ref_out, torch.Tensor):
                # close file
                print(
                    "an output of the forward pass of the reference model is not a tensor"
                )
                temp_file.close()
                return False
            is_close = (
                torch.allclose(ref_out, triton_out),
                "Outputs are not close",
            )

            if not isinstance(triton_out, torch.Tensor):
                # close file
                print(
                    "an output of the forward pass of the triton model is not a tensor"
                )
                temp_file.close()
                return False

            is_close = (
                torch.allclose(ref_out, triton_out),
                "Outputs are not close",
            )

            if not is_close:
                # close file
                temp_file.close()
                return False
        temp_file.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run and check Triton kernel against PyTorch reference"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to parquet dataset file"
    )
    parser.add_argument(
        "--uuid", type=int, required=True, help="UUID of kernel to evaluate"
    )
    parser.add_argument(
        "--pytorch_save_path",
        type=str,
        default=None,
        help="Path to save PyTorch reference code",
    )
    parser.add_argument(
        "--triton_save_path",
        type=str,
        default=None,
        help="Path to save Triton kernel code",
    )
    args = parser.parse_args()

    print("Running with args:", args)

    # Validate dataset path
    if not args.dataset_path.endswith(".parquet"):
        parser.error("dataset_path must be a parquet file")
    if not os.path.exists(args.dataset_path):
        parser.error(f"dataset_path {args.dataset_path} does not exist")

    # Load and filter dataset
    dataset = pd.read_parquet(args.dataset_path)
    # print all uuid values

    curr_row = dataset[dataset["uuid"] == args.uuid]
    if len(curr_row) != 1:
        parser.error(f"Found {len(curr_row)} rows with uuid {args.uuid}, should be 1")

    pytorch_code = curr_row["python_code"].iloc[0]
    triton_code = curr_row["triton_code"].iloc[0]
    entry_point = curr_row["entry_point"].iloc[0]

    # Save code files if paths provided
    if args.pytorch_save_path:
        with open(args.pytorch_save_path, "w") as f:
            f.write(pytorch_code)
    if args.triton_save_path:
        with open(args.triton_save_path, "w") as f:
            f.write(triton_code)

    # # Start Evaluation
    device = torch.device("cuda:0")
    check = evaluate_ref_and_kernel_correctness(
        pytorch_code, triton_code, entry_point, 100
    )
    if check:
        print("Kernel is correct relative to reference")
    else:
        print("Kernel is incorrect relative to reference")


if __name__ == "__main__":
    main()
