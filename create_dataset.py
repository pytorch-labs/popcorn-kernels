import argparse
import glob
import json
import os
import subprocess
from collections import defaultdict

import torch
from compile_code import compile_from_folder
from torch.utils._get_clean_triton import get_clean_triton


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
    for uuid, code_files in uuid_to_code.items():
        if len(code_files) != 1:
            print(f"UUID {uuid} has more than one code file: {code_files}")
            uuid_to_code.pop(uuid)

    # clean the triton code
    for uuid, code_files in uuid_to_code.items():
        code_file = list(code_files)[0]
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"
        subprocess.run(["python", code_file])
        get_clean_triton(code_file, f"cleaned_triton/{uuid}.py")
        uuid_to_clean_code[uuid] = f"cleaned_triton/{uuid}.py"

    print(uuid_to_clean_code)

    for uuid, code_file in uuid_to_clean_code.items():
        generated_code_file = f"generated/random_torch_{uuid}.py"
        with open(generated_code_file, "r") as f:
            generated_code = f.read()
        with open(code_file, "r") as f:
            triton_code = f.read()
        dataset.append(
            {"uuid": uuid, "pytorch_code": generated_code, "triton_code": triton_code}
        )

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
