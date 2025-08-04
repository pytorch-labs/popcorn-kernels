# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import subprocess
from typing import List

import astor


def get_modules_in_testcases(file_content: str) -> List[str]:
	"""
	Given the full content of a Python file, this function returns a list of module names
	that are used in the TESTCASES assignment. These are always the first element of each tuple.
	"""

	tree = ast.parse(file_content)

	test_nodes = []
	modules = []
	for node in tree.body:
		# Look for an assignment to TESTCASES.
		if isinstance(node, ast.Assign) and any(
			isinstance(t, ast.Name) and t.id == 'TESTCASES' for t in node.targets
		):
			test_nodes.append(node)
	if test_nodes:
		test_assign = test_nodes[0]
		if isinstance(test_assign.value, ast.List):
			for elt in test_assign.value.elts:
				if isinstance(elt, ast.Tuple) and elt.elts:
					first_elem = elt.elts[0]
					modules.append(first_elem.id)

	return modules


def extract_target_module(file_content: str, target_module: str) -> str:
	"""
	Given the full content of a Python file and a target module name,
	this function returns a new source code string that contains:
	  - A cleaned header: only import statements from the original header.
	  - Only the target module's class definition and any classes it depends on.
	  - Any helper functions used by the target module.
	  - A filtered TESTCASES assignment that includes only test cases whose first element
	    matches the target module.
	  - A main() function that uses the init and forward lambdas (from the test case)
	    to instantiate the module and print its output.

	Args:
	    file_content (str): The complete source code as a string.
	    target_module (str): The name of the target module (e.g. "BlazeFace").

	Returns:
	    str: A new source code string containing only the desired definitions and a main() function.
	"""
	# --- Step 1. Parse the original file into an AST.
	tree = ast.parse(file_content)

	# --- Step 2. Split the AST into header nodes, class definitions, function definitions and other nodes.
	header_nodes = []
	class_nodes = []
	function_nodes = []
	test_nodes = []
	other_nodes = []
	first_class_encountered = False
	for node in tree.body:
		if isinstance(node, ast.ClassDef):
			first_class_encountered = True
			class_nodes.append(node)
		elif isinstance(node, ast.FunctionDef):
			function_nodes.append(node)
		elif not first_class_encountered:
			header_nodes.append(node)
		else:
			# Look for an assignment to TESTCASES.
			if isinstance(node, ast.Assign) and any(
				isinstance(t, ast.Name) and t.id == 'TESTCASES' for t in node.targets
			):
				test_nodes.append(node)
			else:
				other_nodes.append(node)

	# --- Step 2b. Clean the header to only include import statements.
	header_nodes = [node for node in header_nodes if isinstance(node, (ast.Import, ast.ImportFrom))]

	# --- Step 3. Build a dictionary of top-level classes and record their order.
	class_dict = {}
	order_list = []  # List of (class name, lineno)
	for node in class_nodes:
		class_dict[node.name] = node
		order_list.append((node.name, node.lineno))

	if target_module not in class_dict:
		raise ValueError(f"Target module '{target_module}' not found among class definitions.")

	# --- Step 4. Dependency analysis using AST.
	def find_referenced_names(node):
		"""
		Walks the AST node and returns a set of names that are referenced
		and that are present in our class_dict or are function names.
		"""
		referenced = set()
		for subnode in ast.walk(node):
			if isinstance(subnode, ast.Name):
				if subnode.id in class_dict:
					referenced.add(('class', subnode.id))
				elif subnode.id in [f.name for f in function_nodes]:
					referenced.add(('function', subnode.id))
		return referenced

	# Recursively gather all dependencies starting from the target module.
	selected_classes = set()
	selected_functions = set()
	to_visit = [('class', target_module)]
	while to_visit:
		current_type, current_name = to_visit.pop()
		if current_type == 'class' and current_name in selected_classes:
			continue
		if current_type == 'function' and current_name in selected_functions:
			continue

		if current_type == 'class':
			selected_classes.add(current_name)
			current_node = class_dict[current_name]
		else:
			selected_functions.add(current_name)
			current_node = next(f for f in function_nodes if f.name == current_name)

		refs = find_referenced_names(current_node)
		for ref_type, ref_name in refs:
			if ref_type == 'class' and ref_name not in selected_classes:
				to_visit.append(('class', ref_name))
			elif ref_type == 'function' and ref_name not in selected_functions:
				to_visit.append(('function', ref_name))

	# --- Step 5. Order the selected classes by their original order.
	selected_order = [
		name for name, _ in sorted(order_list, key=lambda x: x[1]) if name in selected_classes
	]

	# --- Step 6. Get the selected functions
	selected_function_nodes = [f for f in function_nodes if f.name in selected_functions]

	# --- Step 7. Reconstruct the header, functions and selected class definitions as source code.
	header_source = ''.join(astor.to_source(node) for node in header_nodes)
	functions_source = '\n\n'.join(
		astor.to_source(node).rstrip() for node in selected_function_nodes
	)
	classes_source = '\n\n'.join(
		astor.to_source(class_dict[name]).rstrip() for name in selected_order
	)

	# --- Step 8. Process the TESTCASES block.
	filtered_elts = []
	if test_nodes:
		test_assign = test_nodes[0]
		if isinstance(test_assign.value, ast.List):
			for elt in test_assign.value.elts:
				if isinstance(elt, ast.Tuple) and elt.elts:
					first_elem = elt.elts[0]
					if isinstance(first_elem, ast.Name) and first_elem.id == target_module:
						filtered_elts.append(elt)
			new_list_node = ast.List(elts=filtered_elts, ctx=ast.Load())
			new_test_assign = ast.Assign(targets=test_assign.targets, value=new_list_node)

	# --- Step 9. Extract the init and forward lambda source strings from the filtered test tuple.
	init_lambda_src = None
	forward_lambda_src = None
	if filtered_elts and len(filtered_elts[0].elts) >= 3:
		init_lambda_src = astor.to_source(filtered_elts[0].elts[1]).strip()
		forward_lambda_src = astor.to_source(filtered_elts[0].elts[2]).strip()

	# --- Step 10. Build a main() function that uses the test case information.
	if init_lambda_src and forward_lambda_src:
		main_function_str = f"""
def get_inputs():
    return ({forward_lambda_src})()

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return ({init_lambda_src})()

"""
	else:
		main_function_str = ''

	# --- Step 11. Reconstruct the final output file content.
	parts = [header_source.rstrip()]
	if functions_source:
		parts.append(functions_source.rstrip())
	parts.append(classes_source.rstrip())
	if main_function_str:
		parts.append(main_function_str.strip())

	new_file_content = '\n\n'.join(parts) + '\n'
	return new_file_content


if __name__ == '__main__':
	# this script will fail so modify these as desired as it's for testing
	new_file_name = 'output.py'
	old_file_name = 'run1/generated_flattened_modules/test_1715labs_baal.py'
	module_name = 'ConsistentDropout2d'
	file1_str = open(old_file_name).read()
	# new_file_str = extract_module(module_name, file1_str)
	new_file_str = extract_target_module(file1_str, module_name)
	print(new_file_str)
	with open(new_file_name, 'w') as f:
		f.write(new_file_str)
	# apply ruff to the file
	subprocess.run([
		'ruff',
		'check',
		new_file_name,
		'--fix',
		'--unsafe-fixes',
		'--fix-only',
	])
	#
	# Similarly, to get the file that tests BCEDiceLoss:
	# new_file_str = extract_module("BCEDiceLoss", file1_str)
	#
	# For demonstration, here we simply print a placeholder message.
	print(
		"This function 'extract_module' can now be used to extract a runnable file for a given module."
	)
