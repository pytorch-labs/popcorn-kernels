# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Util file
"""

import re
import ast, astor
import os
from google import genai
from google.genai import types
from openai import OpenAI

import torch
import concurrent.futures
import time
from tqdm import tqdm

from typing import Any, List, Tuple, Dict, Optional


def generate_gemini(
	prompt: str,
	model: str = 'gemini-2.0-flash',
	temperature: float = 1,
	max_tokens: int = 4096,
	verbose: bool = False,
):
	"""
	Querying Gemini API
	"""
	client = genai.Client(
		api_key=os.environ.get('GEMINI_API_KEY'),
	)
	contents = [
		types.Content(
			role='user',
			parts=[
				types.Part.from_text(text=prompt),
			],
		),
	]

	generate_content_config = types.GenerateContentConfig(
		temperature=temperature,
		top_p=0.95,
		top_k=40,
		max_output_tokens=max_tokens,
		response_mime_type='text/plain',
	)

	# show a select few attributes
	config_dict = {
		'temperature': generate_content_config.temperature,
		'top_p': generate_content_config.top_p,
		'top_k': generate_content_config.top_k,
		'max_output_tokens': generate_content_config.max_output_tokens,
		'response_mime_type': generate_content_config.response_mime_type,
	}
	if verbose:
		print(f'Querying Gemini {model} with config: {config_dict}')

	response = client.models.generate_content(
		model=model,
		contents=contents,
		config=generate_content_config,
	)
	return response.text


def generate_local_server_openai(
	prompt: str,
	model: str = 'define-your-local-model',
	server_address: str = '0.0.0.0',
	port: int = 10210,
	temperature: float = 0,
	max_tokens: int = 100,
	verbose: bool = False,
):
	"""
	For Querying local server (vLLM, SGLang, Tokasaurus, etc.)
	Assume OpenAI Interface, but API key is not needed
	"""
	base_url = f'http://{server_address}:{port}/v1'
	if verbose:
		print(f'Querying {model} with config: {base_url}')
	client = OpenAI(
		api_key='fake-key',  # do not need this
		base_url=base_url,
		# make it very large since local server could be throughput oriented machine, give them enough time for generation
		timeout=100_000,
	)

	response = client.completions.create(
		model='default',
		prompt=prompt,
		temperature=temperature,
		n=1,
		max_tokens=max_tokens,
	)

	outputs = [choice.text for choice in response.choices]

	# print(f"Outputs: {outputs}") # debug
	return outputs[0]


def extract_last_code(output_string: str, code_language_type: str) -> str:
	"""
	Extract the last code block from the output string
	"""
	trimmed = output_string.strip()
	# Extracting all occurrences of content between backticks
	code_matches = re.finditer(r'```(.*?)```', trimmed, re.DOTALL)
	last_match = None

	# Find the last match
	for code_match in code_matches:
		last_match = code_match

	if last_match:
		# Strip leading and trailing whitespace from the extracted code
		code = last_match.group(1).strip()
		# depends on code_language_type: cpp, python, etc.
		# sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
		# in this case strip the cpp out
		if code.startswith(code_language_type):
			code = code[len(code_language_type) :].strip()
		return code
	return None


def extract_final_pattern(output_string: str) -> list[str]:
	pattern_match = re.search(r'Final Pattern: \[(.*?)\]', output_string)
	if not pattern_match:
		return None

	final_pattern = [op.strip().strip("'") for op in pattern_match.group(1).split(',')]

	return final_pattern


def swap_forward_call(pytorch_code, entry_point, generate_forward_call):
	"""
	Sahan's AST based swapping code (did not test this yet)
	"""
	# find the forward call and remove it and return the rest of the code
	tree = ast.parse(pytorch_code)

	# Helper function to rename Name nodes
	def rename_references(node):
		for child_node in ast.walk(node):
			if isinstance(child_node, ast.Name) and child_node.id == entry_point:
				child_node.id = f'{entry_point}New'

	for node in ast.walk(tree):
		# find the module definition for the entry point
		if isinstance(node, ast.ClassDef) and node.name == entry_point:
			# Rename the class definition
			node.name = f'{entry_point}New'

			# Scan all nodes in the class body to replace references to entry_point
			for child in node.body:
				rename_references(child)

				# Additionally, replace the forward method
				if isinstance(child, ast.FunctionDef) and child.name == 'forward':
					node.body.remove(child)
					# replace with the generate_forward_call function with proper indentation
					node.body.append(ast.parse(generate_forward_call).body[0])

	# Also scan for references outside the class definition
	for node in ast.walk(tree):
		if isinstance(node, ast.Name) and node.id == entry_point:
			node.id = f'{entry_point}New'

	return astor.to_source(tree)


def num_generations_in_dir(dir_path: os.PathLike) -> int:
	"""
	Count the number of files in the directory
	"""
	return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])


def test_synthetic_model(torch_src: str, entry_point: str) -> Tuple[bool, Optional[str]]:
	"""
	Tests a synthetic model by loading it with exec, initializing it with get_init_inputs,
	and running it with get_inputs.
	Check if the model can pass torch Eager and torch.compile
	Returns a tuple of (success, error_message)
	"""
	try:
		# Read and execute the file content
		namespace = {}
		exec(torch_src, namespace)

		# Find the model class by name
		model_class = namespace.get(entry_point)
		if not model_class or not issubclass(model_class, torch.nn.Module):
			return False, f'Could not find valid PyTorch model class named {entry_point}'

		# Get initialization parameters
		get_init_inputs = namespace.get('get_init_inputs')
		if not get_init_inputs:
			return False, 'get_init_inputs function not found in the file'

		init_args, init_kwargs = get_init_inputs()

		# Initialize the model
		model = model_class(*init_args, **init_kwargs)

		# Get input tensors
		get_inputs = namespace.get('get_inputs')
		if not get_inputs:
			return False, 'get_inputs function not found in the file'

		inputs = get_inputs()

		# Test eager mode
		with torch.no_grad():
			try:
				outputs = model(*inputs)

				# Basic validation of outputs
				if not isinstance(outputs, (torch.Tensor, tuple, list)):
					return False, 'Model output is not a tensor or tuple/list of tensors'
			except Exception as e:
				return False, f'Error during eager mode forward pass: {str(e)}'

		# Test compiled mode
		try:
			# default Inductor backend
			compiled_model = torch.compile(model)
			compiled_outputs = compiled_model(*inputs)

			# Basic validation of compiled outputs
			if not isinstance(compiled_outputs, (torch.Tensor, tuple, list)):
				return False, 'Compiled model output is not a tensor or tuple/list of tensors'

			return True, None
		except Exception as e:
			return False, f'Error in compiled mode: {str(e)}'

	except Exception as e:
		return False, f'Error: {str(e)}'


def maybe_multithread(
	func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs
):
	"""
	From KernelBench repo
	Multithreaded execution of func, with optional time interval between queries
	Ideal for querying LLM APIs, does not provide process isolation
	"""
	output_data = []
	if num_workers not in [1, None]:
		with tqdm(total=len(instances), smoothing=0) as pbar:
			with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
				# Submit tasks one at a time with delay between them
				futures = []
				for instance in instances:
					futures.append(executor.submit(func, instance, *shared_args, **shared_kwargs))
					time.sleep(time_interval)  # sleep between submitting each task

				# Wait for each future to complete
				for future in concurrent.futures.as_completed(futures):
					pbar.update(1)
					try:
						result = future.result()
						if result is not None:
							output_data.append(result)
					except Exception as e:
						print('Got an error!', e)
						continue
	else:
		for instance in tqdm(instances):
			output = func(instance, *shared_args, **shared_kwargs)
			if output is not None:
				output_data.append(output)

	return output_data


def maybe_multiprocess(
	func, instances, num_workers, time_interval=0.0, *shared_args, **shared_kwargs
):
	"""
	From KernelBench repo
	Multiprocessed execution of func,
	Ideal for querying LLM APIs, bu this one also provides process isolation

	TODO: add optional time interval between queries
	"""
	output_data = []
	if num_workers not in [1, None]:
		with tqdm(total=len(instances), smoothing=0) as pbar:
			with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
				# Create a future for running each instance
				futures = {
					executor.submit(func, instance, *shared_args, **shared_kwargs): None
					for instance in instances
				}
				# Wait for each future to complete
				for future in concurrent.futures.as_completed(futures):
					pbar.update(1)
					try:
						result = future.result()
						if result is not None:
							output_data.append(result)
					except Exception as e:
						print('Got an error!', e)
						continue

	else:
		for instance in tqdm(instances):
			output = func(instance, *shared_args, **shared_kwargs)
			if output is not None:
				output_data.append(output)

	return output_data


def main():
	generate_local_server_openai(
		prompt='Hello, world!',
		model='gpt-4o-mini',
		server_address='matx2.stanford.edu',
		port=10210,
		temperature=0.7,
		max_tokens=10,
		verbose=True,
	)


if __name__ == '__main__':
	main()

# # Example usage:
# if __name__ == "__main__":
#     # Example file path (adjust according to your project structure)
#     file_path = os.path.join(os.getcwd(), "synth_torch_debug/synth_torch_Conv2d_AvgPool2d.py")
#     # Ensure file exists before testing
#     print(f"Current working directory: {os.getcwd()}")
#     assert os.path.exists(file_path), f"Error: File {file_path} does not exist"
#     entry_point = "SynthModel_Conv2d_AvgPool2d"
#     with open(file_path, 'r') as f:
#         file_content = f.read()
#     success, error = test_synthetic_model(file_content, entry_point)
#     if success:
#         print("Model test successful!")
#     else:
#         print(f"Model test failed: {error}")
