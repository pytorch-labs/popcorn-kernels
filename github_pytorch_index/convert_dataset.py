# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert a dataset file into
dataset format for Llama Factory (Alpaca)
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
  }
]
"""

import json
import argparse

from tqdm import tqdm

TRITON_INSTRUCTION = (
	'Convert this PyTorch neural network code to its optimized Triton implementation.'
)


def convert_dataset(input_file, output_file):
	# Read the input JSON file
	with open(input_file, 'r') as f:
		data = json.load(f)

	# Convert to Alpaca format
	converted_data = []
	for item in tqdm(data):
		converted_item = {
			'instruction': TRITON_INSTRUCTION,
			'input': item['python_code'],
			'output': item['triton_code'],
		}
		converted_data.append(converted_item)

	# Write to output file
	with open(output_file, 'w') as f:
		json.dump(converted_data, f, indent=2)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file', help='Input dataset file path')
	parser.add_argument('--output_file', help='Output dataset file path', default=None)
	args = parser.parse_args()

	# Set default output file if not provided
	output_file = args.output_file or args.input_file.replace('.json', '_alpaca.json')
	convert_dataset(args.input_file, output_file)
