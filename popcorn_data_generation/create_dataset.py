import argparse
import glob
import io
import json
import os
import subprocess
import tokenize
from collections import defaultdict
from io import BytesIO

import tqdm
from compile_code import compile_from_folder
from torch.utils._get_clean_triton import get_clean_triton


def remove_python_comments(source: str) -> str:
    """
    Remove all comments from a Python source code string without altering other formatting.

    This function uses the built-in tokenize module to break the source code
    into tokens, then reconstructs the source code while omitting any token
    of type COMMENT. It carefully adds back any whitespace or newlines that occur
    between tokens, so that the formatting of the remaining code is preserved.
    """

    # Encode the source to bytes and create a BytesIO stream for tokenization.
    source_bytes = source.encode("utf-8")
    stream = BytesIO(source_bytes)

    # Initialize the token generator.
    tokens = tokenize.tokenize(stream.readline)

    # We'll rebuild the source using pieces accumulated in this list.
    result = []
    # Keep track of the position (line, column) of the end of the last token added.
    last_lineno, last_col = 1, 0

    for token in tokens:
        token_type = token.type
        token_string = token.string
        start_line, start_col = token.start
        end_line, end_col = token.end

        # Skip the encoding and endmarker tokens.
        if token_type in (tokenize.ENCODING, tokenize.ENDMARKER):
            continue

        if token_type == tokenize.COMMENT:
            # Instead of outputting the comment, update the current position.
            # This has the effect of “removing” the comment along with any space that was
            # solely part of the comment region.
            last_lineno, last_col = end_line, end_col
            continue

        # If there is a gap between the last token and the current token,
        # fill it in (this preserves spaces and newlines from the original source).
        if start_line > last_lineno:
            # Add newlines for any skipped lines.
            # result.append("\n")
            # After a newline, reset column to 0.
            last_col = 0

        # Add any extra spaces needed to reach the token’s start column.
        if start_col > last_col:
            result.append(" " * (start_col - last_col))

        # Append the current token’s text.
        result.append(token_string)
        # Update the last position to the end of the current token.
        last_lineno, last_col = end_line, end_col

    # Join all pieces and return the reconstructed source.
    return "".join(result)


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
        # code_file = list(code_files)[0]
        # if not os.path.exists(f"cleaned_triton/{uuid}.py"):
        #     try:
        #         os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"
        #         subprocess.run(["python", code_file])
        #         get_clean_triton(code_file, f"cleaned_triton/{uuid}.py")
        #     except Exception as e:
        #         print(f"Failed to clean triton code for {uuid}: {e}")
        #         continue
        # else:
        #     # print(f"cleaned_triton/{uuid}.py already exists, skipping")
        #     pass
        uuid_to_clean_code[uuid] = f"cleaned_triton/{uuid}.py"

    # copy cleaned triton code to linted folder
    # create the linted_triton folder if it doesn't exist
    if not os.path.exists("linted_triton"):
        os.makedirs("linted_triton")
    # copy the cleaned triton code to the linted_triton folder
    # clean out the linted_triton folder first
    subprocess.run(["rm", "-rf", "linted_triton/*"])
    # lint the triton code
    # for uuid, code_file in tqdm.tqdm(
    #     uuid_to_clean_code.items(), desc="cleaning dataset"
    # ):
    bad_files = []
    for uuid in tqdm.tqdm(uuid_to_clean_code.keys(), desc="linting code"):
        try:
            original_code_file = f"cleaned_triton/{uuid}.py"
            linted_code_file = f"linted_triton/{uuid}.py"
            if not os.path.exists(original_code_file):
                continue
            code = open(original_code_file, "r").read()
            commentless_code = remove_python_comments(code)
            with open(linted_code_file, "w") as f:
                f.write(commentless_code)
        except Exception as e:
            # print(f"Failed to clean triton code for {uuid}: {e}")
            bad_files.append(uuid)
    # apply ruff linter
    subprocess.run(
        [
            "ruff",
            "check",
            "linted_triton",
            "--fix",
            "--unsafe-fixes",
            "--fix-only",
        ]
    )
    for uuid in tqdm.tqdm(uuid_to_clean_code.keys(), desc="Creating dataset"):
        code_file = f"linted_triton/{uuid}.py"
        if not os.path.exists(code_file):
            continue
        else:
            generated_code_file = f"generated/random_torch_{uuid}.py"
            with open(generated_code_file, "r") as f:
                generated_code = f.read()
            with open(code_file, "r") as f:
                triton_code = f.read()
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
