import argparse
import glob
import io
import json
import os
import subprocess
import tokenize
from collections import defaultdict

import tqdm
from compile_code import compile_from_folder
from torch.utils._get_clean_triton import get_clean_triton


def remove_extraneous_newlines(file_path):
    """
    Read the lines from the given Python file, and remove any "extra" blank lines
    so that consecutive blank lines collapse into just one.

    1. Read all lines from the file.
    2. Track whether the previous line was blank.
    3. Only append a blank line to the new list of lines if the previous line wasn't blank.
    4. Return the adjusted code as a string.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    prev_line_blank = False

    for line in lines:
        if line.strip() == "":
            # This line is blank
            if not prev_line_blank:
                # Keep one blank line if the previous line wasn't blank
                cleaned_lines.append(line)
            # Mark that we've encountered a blank line
            prev_line_blank = True
        else:
            # This line isn't blank, so just keep it
            cleaned_lines.append(line)
            # Reset the blank line marker
            prev_line_blank = False

    return "".join(cleaned_lines)


def remove_python_comments(file_path):
    """
    Removes all single-line and inline Python comments from the given file.

    Steps:
    1. Read the source file into a string.
    2. Tokenize the string using Python's built-in tokenize.generate_tokens.
    3. Filter out any token of type COMMENT.
    4. Use tokenize.untokenize to reconstruct valid Python code from the filtered tokens.

    Returns:
        A string containing the original Python code without comments.
    """
    with open(file_path, "r") as f:
        code = f.read()

    # Tokenize the file's content
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)

    # Filter out COMMENT tokens
    filtered_tokens = []
    for token_type, token_string, start_pos, end_pos, line in tokens:
        if token_type == tokenize.COMMENT:
            # Skip any token identified as a comment
            continue
        filtered_tokens.append((token_type, token_string))

    # Reconstruct the code without comments
    return tokenize.untokenize(filtered_tokens)


def extract_output_code(dir_path):
    # read the file line by line and look for lines that look like
    # [stuff] Output code written to: /tmp/torchinductor_sahanp/y4/cy4ujkrhmfu3t5xkvy53nswxoy5b7he2246t2rrxutae4ksl3dfe.py
    uuid_to_code = defaultdict(set)
    dataset = []
    uuid_to_clean_code = {}
    for file_path in glob.glob(os.path.join(dir_path, "*.txt")):
        with open(file_path, "r") as f:
            for line in f:
                if "Output code written to:" in line:
                    uuid = file_path.split("_")[-1].split(".")[0]
                    code_file = line.split("Output code written to:")[1].strip()
                    uuid_to_code[uuid].add(code_file)
    # flag the ones that have more than one code file
    bad_uuids = set()
    print(f"Found {len(uuid_to_code)} entries")
    for uuid, code_files in uuid_to_code.items():
        if len(code_files) != 1:
            print(f"UUID {uuid} has more than one code file: {code_files}")
            bad_uuids.add(uuid)
    print(f"Found {len(bad_uuids)} bad uuids")
    print(f"Found {len(uuid_to_code)} entries before filtering")
    for uuid in bad_uuids:
        uuid_to_code.pop(uuid)
    print(f"Found {len(uuid_to_code)} entries after filtering")
    # clean the triton code
    for uuid, code_files in tqdm.tqdm(
        uuid_to_code.items(), desc="Cleaning triton code"
    ):
        code_file = list(code_files)[0]
        if not os.path.exists(f"cleaned_triton/{uuid}.py"):
            continue
            try:
                os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"
                subprocess.run(["python", code_file])
                get_clean_triton(code_file, f"cleaned_triton/{uuid}.py")
            except Exception as e:
                print(f"Failed to clean triton code for {uuid}: {e}")
                continue
        else:
            print(f"cleaned_triton/{uuid}.py already exists, skipping")
        uuid_to_clean_code[uuid] = f"cleaned_triton/{uuid}.py"

    # for uuid, code_file in tqdm.tqdm(
    #     uuid_to_clean_code.items(), desc="cleaning dataset"
    # ):
    bad_files = []
    for uuid, code_file in tqdm.tqdm(uuid_to_clean_code.items(), desc="linting code"):
        try:
            commentless_code = remove_python_comments(code_file)
            with open(code_file, "w") as f:
                f.write(commentless_code)
            clean_code = remove_extraneous_newlines(code_file)
            with open(code_file, "w") as f:
                f.write(clean_code)
            # remove unused imports and variables

            subprocess.run(
                [
                    "autoflake",
                    "--in-place",
                    "--remove-all-unused-imports",
                    "--remove-unused-variables",
                    code_file,
                ]
            )
        except Exception as e:
            # print(f"Failed to clean triton code for {uuid}: {e}")
            bad_files.append(uuid)

    for uuid, code_file in tqdm.tqdm(
        uuid_to_clean_code.items(), desc="Creating dataset"
    ):
        if uuid in bad_files:
            continue
        else:
            generated_code_file = f"generated/random_torch_{uuid}.py"
            with open(generated_code_file, "r") as f:
                generated_code = f.read()
            with open(code_file, "r") as f:
                triton_code = f.read()
            with open(code_file, "w") as f:
                f.write(triton_code)
            dataset.append(
                {
                    "uuid": uuid,
                    "pytorch_code": generated_code,
                    "triton_code": triton_code,
                }
            )
    print(f"Found {len(dataset)} entries")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir_path", type=str, default="generated")
    parser.add_argument("--output_file", type=str, default="dataset.json")
    parser.add_argument("--uuid_file", type=str, default="filtered_uuids.json")
    args = parser.parse_args()

    intermediate_output_path = "inductor_dump"
    compile_from_folder(args.gen_dir_path, args.uuid_file, intermediate_output_path)

    dataset = extract_output_code(intermediate_output_path)
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
