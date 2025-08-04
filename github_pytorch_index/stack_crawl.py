# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from datasets import load_dataset

# Use streaming to handle the large dataset efficiently
ds = load_dataset(
    "bigcode/the-stack-dedup", data_dir="data/python", streaming=True, split="train"
)

# Track unique repositories containing torch imports
torch_repos = set()
repo_dicts = []

# Regular expression to match different torch import patterns
torch_import_strings = [
    "import torch",
    "from torch import",
]

# Process the dataset
for sample in iter(ds):
    content = sample["content"]
    repo_name = sample["max_stars_repo_name"]
    lang = sample["lang"]
    sha = sample["max_stars_repo_head_hexsha"]
    licenses = sample["max_stars_repo_licenses"]
    stars = sample["max_stars_count"]

    if lang != "Python" or repo_name in torch_repos:
        continue
    index = 0
    # Check if file contains torch import
    for torch_import_string in torch_import_strings:
        if torch_import_string in content:
            repo_dicts.append(
                {
                    "sha": sha,
                    "licenses": licenses,
                    "stars": stars,
                    "repo_name": repo_name,
                }
            )
            index += 1
            torch_repos.add(repo_name)
            # Optional: Print progress every N repositories
            if len(torch_repos) % 1000 == 0 and len(torch_repos) > 0:
                print(
                    f"Current count of repositories with torch imports: {len(torch_repos)}"
                )
            break

try:
    # sort the repo_dicts by stars - handle None values by treating them as -1
    sorted_repo_dicts = sorted(repo_dicts, key=lambda x: x["stars"] if x["stars"] is not None else -1, reverse=True)
    for i, repo in enumerate(sorted_repo_dicts):
        repo["index"] = i
    # write the sorted repo_dicts to a json file
    json.dump(sorted_repo_dicts, open("torch_repos.json", "w"), indent=4)

    print(f"\nFinal count of repositories containing torch imports: {len(torch_repos)}")
except Exception as e:
    print(f"Error: {e}")
    json.dump(repo_dicts, open("torch_repos.json", "w"), indent=4)
