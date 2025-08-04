#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
import tqdm


def try_compile(input_file_path):
	"""
	Reads a Python file assumed to define a callable named 'model',
	compiles it with torch.compile, and dumps the generated Triton code.
	The code is then saved to the given output file path.
	"""

	# Read and execute the Python source file in an isolated dict
	# sys.stdout = open(output_file_path, "w")
	# sys.stderr = open(output_file_path, "w")
	local_dict = {}
	with open(input_file_path, 'r') as f:
		code = f.read()
	exec(code, local_dict)

	# Retrieve the model/function and compile it with torch.compile
	# (Adjust 'model' to the actual name if needed)
	model = local_dict.get('Model', None)
	init_inputs = local_dict.get('get_init_inputs', None)
	dummy_init_inputs = []
	dummy_input = []
	if init_inputs is not None:
		dummy_init_inputs = init_inputs()
	input_fn = local_dict.get('get_inputs', None)
	if input_fn is not None:
		dummy_input = input_fn()

	if model is None:
		print(f"No 'model' found in {input_file_path}. Skipping.")
		return
	try:
		# todo: make inputs work for tuples
		dummy_init_inputs = [x.cuda() for x in dummy_init_inputs]
		dummy_input = [x.cuda() for x in dummy_input]

		model = model(*dummy_init_inputs)
		model = model.cuda()
		compiled_model = torch.compile(model, backend='inductor')

		# Run the compiled model to ensure Triton code is actually generated.
		# Here, we just call it with a random tensor as an example.
		_ = compiled_model(*dummy_input)
	except Exception as e:
		print(f'Error compiling {input_file_path}: {e}')
		return


def compile_from_folder(gen_folder, uuid_file, output_folder='inductor_dump'):
	# Grab all files in the 'generated' folder that match random_torch_{uuid}.py
	valid_uuids = json.load(open(uuid_file, 'r'))
	for uuid in tqdm.tqdm(valid_uuids, desc='Compiling files'):
		file_path = f'{gen_folder}/random_torch_{uuid}.py'
		output_file_path = f'{output_folder}/generated_torch_compiled_{uuid}.txt'
		if os.path.exists(output_file_path):
			print(f'Skipping {file_path} because {output_file_path} already exists.')
			continue
		os.environ['TORCH_LOGS_OUT'] = output_file_path
		torch._logging.set_logs(output_code=True)
		try_compile(file_path)
