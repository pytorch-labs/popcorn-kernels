# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast

import astor


def transform_get_functions(code_str):
	"""
	Transforms get_inputs and get_init_inputs functions to return just the first list
	from the tuple instead of using a lambda function.

	Args:
	    code_str (str): String containing Python code with get_inputs and get_init_inputs functions

	Returns:
	    str: Modified code with simplified return statements
	"""
	# Parse the code into an AST
	tree = ast.parse(code_str)

	# Find and transform the target functions
	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef) and node.name in [
			'get_inputs',
			'get_init_inputs',
		]:
			# For each statement in the function body
			for i, stmt in enumerate(node.body):
				# If it's a return statement
				if isinstance(stmt, ast.Return):
					# Try to find a Lambda node
					lambda_node = None
					for subnode in ast.walk(stmt.value):
						if isinstance(subnode, ast.Lambda):
							lambda_node = subnode
							break

					# If a Lambda was found
					if lambda_node:
						# Check if its body is a Tuple
						if isinstance(lambda_node.body, ast.Tuple) and lambda_node.body.elts:
							# Extract the first element (which is the list we want)
							first_element = lambda_node.body.elts[0]
							second_element = lambda_node.body.elts[1]

							# Replace the return statement with just that list if it's forward
							if node.name == 'get_inputs':
								node.body[i] = ast.Return(value=first_element)
							elif node.name == 'get_init_inputs':
								# Replace the return statement with a List of the first and second elements
								node.body[i] = ast.Return(
									value=ast.List(
										elts=[first_element, second_element],
										ctx=ast.Load(),
									)
								)
	# Convert the modified AST back to source code
	return astor.to_source(tree)
