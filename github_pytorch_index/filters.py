# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this file is for all the filters we stick on for examples with the added benefit of we track things using the produced files.
import ast

from create_triton_data import parse_assert_size_stride, run_model


def check_non_functional(python_code, triton_code, entry_point):
	# if the number of params and inputs is the same as the number of primals, then it is functional
	tensor_info = parse_assert_size_stride(triton_code)
	params_dict, inputs_dict = run_model(python_code, entry_point)
	return len(params_dict) + len(inputs_dict) == len(tensor_info)


def check_uses_kwargs(python_code, entry_point):
	# as kernel bench does not use forward kwargs, we will filter out models which use them in forward
	context = {}
	compile(python_code, '<string>', 'exec')
	exec(python_code, context)
	get_inputs = context['get_inputs']

	_, forward_kwargs = get_inputs()
	return len(forward_kwargs) > 0


def check_single_return(pytorch_code, entry_point):
	# Parse the code into an AST
	tree = ast.parse(pytorch_code)

	# Find the module definition for the entry point
	target_class = None
	for node in ast.walk(tree):
		if isinstance(node, ast.ClassDef) and node.name == entry_point:
			target_class = node
			break

	if not target_class:
		raise ValueError(f"Class '{entry_point}' not found in the code.")

	# Find the forward method in the class
	forward_method = None
	for child in target_class.body:
		if isinstance(child, ast.FunctionDef) and child.name == 'forward':
			forward_method = child
			break

	if not forward_method:
		raise ValueError(f"No 'forward' method found in class '{entry_point}'.")

	# Find all return statements in the forward method
	return_nodes = []
	for node in ast.walk(forward_method):
		if isinstance(node, ast.Return):
			return_nodes.append(node)
	return len(return_nodes) == 1
