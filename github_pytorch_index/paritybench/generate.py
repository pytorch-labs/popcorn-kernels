# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import shutil
import sys
import time
import types
from functools import partial
from multiprocessing.pool import ThreadPool
from unittest.mock import patch

from code_transforms import transform_get_functions
from extract import extract_target_module, get_modules_in_testcases
from filters import check_uses_kwargs

from paritybench.module_extractor import PyTorchModuleExtractor
from paritybench.reporting import ErrorAggregatorDict, Stats
from paritybench.utils import subproc_wrapper
from tqdm import tqdm
from utils import lint_code_directory

log = logging.getLogger(__name__)


def write_helpers(run_dir):
    """
    Sets up and loads helper functions for paritybench.

    This function:
    1. Creates a symbolic link from generated/_paritybench_helpers.py to the source helpers
    2. Dynamically loads the helper code as a Python module
    3. Makes the helpers available globally via sys.modules

    The function uses symbolic links to maintain a connection between source and generated
    code while keeping them in separate directories. It also handles special cases during
    module loading by temporarily modifying sys.argv.

    """

    src = os.path.join(os.path.dirname(__file__), "_paritybench_helpers.py")
    # src = "paritybench/_paritybench_helpers.py"
    tests_dir = run_dir + "/generated_flattened_modules"
    dst = f"{tests_dir}/_paritybench_helpers.py"

    # Create generated directory if it doesn't exist
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    os.path.exists(dst) and os.unlink(dst)
    os.symlink(os.path.join("..", src), dst)
    helpers_code = open(dst).read()
    with patch("sys.argv", sys.argv[:1]):  # testcase import does annoying stuff
        helpers = types.ModuleType("_paritybench_helpers")
        exec(
            compile(helpers_code, f"{tests_dir}/_paritybench_helpers.py", "exec"),
            helpers.__dict__,
            helpers.__dict__,
        )
        sys.modules["_paritybench_helpers"] = helpers


def generate_zipfile_subproc(tempdir: str, path: str, args):
    """
    Args:
        tempdir: temporary dir
        path: input path process a .zip file from a github download
    """
    errors = ErrorAggregatorDict(path)
    stats = Stats()
    test_dir = args.run_dir + "/generated_flattened_modules"
    test_path = "{}/test_{}.py".format(
        test_dir, re.sub(r"([.]zip|/)$", "", os.path.basename(path))
    )

    if os.path.exists(test_path):
        return errors, stats

    with open(test_path, "w") as output_py:
        extractor = PyTorchModuleExtractor(
            tempdir, errors, stats, output_py=output_py, args=args
        )
        extractor.main(path)
    # grab all files from the test directory

    seperated_module_directory = args.run_dir + "/cleaned_pytorch_modules"
    kwargs_directory = args.run_dir + "/cleaned_pytorch_modules_with_kwargs"
    os.makedirs(seperated_module_directory, exist_ok=True)
    test_file_code = open(test_path, "r").read()
    modules = get_modules_in_testcases(test_file_code)
    repo_name = test_path.split("test_")[-1].split(".py")[0]
    for module in modules:
        extracted_code = extract_target_module(test_file_code, module)
        file_name = f"{repo_name}.{module}.py"
        if check_uses_kwargs(extracted_code, module):
            target_file = os.path.join(kwargs_directory, file_name)
        else:
            # transform get functions in this case as well
            extracted_code = transform_get_functions(extracted_code)
            target_file = os.path.join(seperated_module_directory, file_name)
        with open(target_file, "w") as f:
            f.write(extracted_code)
    return errors, stats


def generate_all(
    args, download_dir, limit=None, jobs=4, chunk_num=None, num_chunks=None
):
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    zipfiles = [
        os.path.join(download_dir, f)
        for f in os.listdir(download_dir)
        if f.endswith(".zip")
    ]
    zipfiles.sort()
    test_dir = args.run_dir + "/generated_flattened_modules"
    python_module_dir = args.run_dir + "/cleaned_pytorch_modules"
    python_with_kwargs_dir = args.run_dir + "/cleaned_pytorch_modules_with_kwargs"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(python_module_dir, exist_ok=True)
    os.makedirs(python_with_kwargs_dir, exist_ok=True)
    print(f"Generating tests for {len(zipfiles)} zip files with {jobs} jobs")

    fgen = partial(generate_zipfile_subproc, args=args)
    generate_zipfile = partial(subproc_wrapper, fn=fgen)
    if limit is not None and (num_chunks is not None or chunk_num is not None):
        raise ValueError("Cannot specify both limit and chunk num/num_chunks")
    if num_chunks is None != chunk_num is None:
        raise ValueError("Must specify both num_chunks and chunk_num or neither")

    if limit:
        zipfiles = zipfiles[:limit]
    if num_chunks is not None or chunk_num is not None:
        num_zipfiles = len(zipfiles)
        chunk_size = num_zipfiles // num_chunks
        zipfiles = zipfiles[chunk_size * chunk_num : chunk_size * (chunk_num + 1)]

    pool = ThreadPool(jobs)
    for errors_part, stats_part in tqdm(
        pool.imap_unordered(generate_zipfile, zipfiles),
        total=len(zipfiles),
        desc="Processing zip files",
    ):
        errors.update(errors_part)
        stats.update(stats_part)
    pool.close()

    lint_code_directory(python_module_dir)

    errors.print_report()
    log.info(f"TOTAL: {stats}, took {time.time() - start:.1f} seconds")
