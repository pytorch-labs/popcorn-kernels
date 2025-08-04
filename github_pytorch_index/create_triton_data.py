# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast

import astor
from utils import run_ruff_on_code


def is_docstring(node):
    """Check if a node is a docstring."""
    if not isinstance(node, ast.Expr):
        return False

    return isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def move_imports_to_top(code_string):
    """
    Moves all import statements to the top of the given Python code string.

    Note: This function will NOT preserve comments in the original code.

    Args:
        code_string (str): A string containing Python code

    Returns:
        str: The modified code with all imports at the top
    """
    # Parse the input code string into an AST
    tree = ast.parse(code_string)

    # Lists to store different types of statements
    future_imports = []
    regular_imports = []
    other_statements = []
    docstring = []

    # Check for module docstring
    if len(tree.body) > 0 and is_docstring(tree.body[0]):
        docstring = [tree.body[0]]
        nodes_to_process = tree.body[1:]
    else:
        nodes_to_process = tree.body

    # Categorize statements
    for node in nodes_to_process:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            future_imports.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            regular_imports.append(node)
        else:
            other_statements.append(node)

    # Create a new AST with statements in the correct order
    new_body = docstring + future_imports + regular_imports + other_statements

    new_tree = ast.Module(body=new_body, type_ignores=[])

    # Convert the modified AST back to source code
    return astor.to_source(new_tree)


def remove_unused_torch_ops(code_string):
    """
    Analyzes Python code and removes torch-related variable assignments
    if those variables are not used elsewhere in the code.

    Args:
        code_string (str): A string containing Python code

    Returns:
        str: The modified code with unused assignments removed
    """
    # Parse the input code string into an AST
    tree = ast.parse(code_string)

    # Find all torch-related assignments and track assigned variables
    assignment_indices = []
    assigned_vars = {}

    for idx, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
                value = node.value

                # Check if it's a torch-related assignment
                # we'll let the linter handle the rest
                if _is_torch_related(value):
                    assignment_indices.append(idx)
                    assigned_vars[var_name] = idx

    # Create a visitor to track variable usages
    class NameUsageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.used_vars = set()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in assigned_vars:
                self.used_vars.add(node.id)
            self.generic_visit(node)

    # Find all variable usages in the code
    visitor = NameUsageVisitor()

    # Visit all nodes to find variable usages
    for idx, node in enumerate(tree.body):
        if idx not in assignment_indices:
            # In non-assignment nodes, visit the whole node
            visitor.visit(node)
        else:
            # In assignment nodes, only visit the value part
            # This avoids counting the target variable as used
            visitor.visit(node.value)

    # Identify unused variables
    unused_indices = []
    for var_name, idx in assigned_vars.items():
        if var_name not in visitor.used_vars:
            unused_indices.append(idx)

    # Create new tree without unused assignments
    new_body = [node for idx, node in enumerate(tree.body) if idx not in unused_indices]

    # Create a new module with the filtered body
    new_tree = ast.Module(body=new_body, type_ignores=[])

    # Convert the modified AST back to source code
    return astor.to_source(new_tree)


def _is_torch_related(node):
    """
    Check if a node is a torch-related operation.
    This handles both direct attribute access and parenthesized expressions.
    """
    if isinstance(node, ast.Attribute):
        base = node
        while isinstance(base, ast.Attribute):
            base = base.value

        return isinstance(base, ast.Name) and base.id == "torch"

    return False


def parse_assert_size_stride(triton_code):
    """Parse assert_size_stride calls to map tensor names to their shapes."""
    # Basic parsing using ast
    tree = ast.parse(triton_code)
    tensor_info = {}

    # there will be an assignment that looks like
    # primal0, primal1, etc = arg
    # we only care about the primal0, primal1, etc

    class AssertVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "assert_size_stride":
                # First arg is tensor name, second arg is shape tuple
                if len(node.args) >= 2:
                    tensor_name = node.args[0].id
                    # Extract shape from the AST
                    if isinstance(node.args[1], ast.Tuple):
                        shape = []
                        for elt in node.args[1].elts:
                            if isinstance(elt, ast.Constant):
                                shape.append(elt.value)
                        tensor_info[tensor_name] = shape
            self.generic_visit(node)

    AssertVisitor().visit(tree)

    filtered_tensor_info = {}
    for tensor_name in tensor_info.keys():
        # args should only be in the form of primals_0, primals_1, etc or arg0, arg1, etc
        if tensor_name.startswith("primals_") or tensor_name.startswith("arg"):
            filtered_tensor_info[tensor_name] = tensor_info[tensor_name]

    return filtered_tensor_info


def write_call_wrapper(triton_code, params_dict, input_dict, num_outputs=1):
    # todo: either figure out kwargs or throw out things with it

    # Parse the assert_size_stride calls to get tensor shapes
    tensor_info = parse_assert_size_stride(triton_code)
    # Match parameters to tensor names based on shapes
    param_mapping = {}
    used_tensor_names = set()

    for param_name, param_data in params_dict.items():
        for tensor_name, expected_shape in tensor_info.items():
            if (
                "shape" in param_data
                and param_data["shape"] == expected_shape
                and param_name not in param_mapping
                and tensor_name not in used_tensor_names
            ):
                param_mapping[tensor_name] = {
                    "param_name": param_data["callable_name"],
                }
                used_tensor_names.add(tensor_name)
                break

    for param_name, param_data in input_dict.items():
        for tensor_name, expected_shape in tensor_info.items():
            if (
                "shape" in param_data
                and param_data["shape"] == expected_shape
                and param_name not in param_mapping
                and tensor_name not in used_tensor_names
            ):
                param_mapping[tensor_name] = {
                    "param_name": param_name.replace(".", "_"),
                }
                used_tensor_names.add(tensor_name)
                break
    input_names = list(input_dict.keys())
    # convert to string separated by commas
    input_names = ", ".join(input_names)
    # Create the wrapper function
    wrapper = f"""

def forward(self, {input_names}):
"""

    for tensor_name, mapping in param_mapping.items():
        wrapper += f"    {tensor_name} = {mapping['param_name']}\n"
    # Add the call to the original triton function
    wrapper += "\n    # Call the Triton kernel\n"
    wrapper += "    output = call(["

    # Add parameters to the call in the order they appear in tensor_info
    for tensor_name in tensor_info.keys():
        if tensor_name in param_mapping:
            wrapper += f"\n        {tensor_name},"

    wrapper += "\n    ])\n\n"

    wrapper += "    return output[0]"
    for i in range(1, num_outputs):
        wrapper += f" , output[{i}]"

    return wrapper


def run_model(file_code, entry_point):
    context = {}
    compile(file_code, "<string>", "exec")
    exec(file_code, context)
    model_class = context[entry_point]
    get_inputs = context["get_inputs"]
    get_init_inputs = context["get_init_inputs"]

    forward_args = get_inputs()
    init_args, init_kwargs = get_init_inputs()
    model = model_class(*init_args, **init_kwargs)

    params_dict = {}
    inputs_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = {
            "callable_name": f"self.{name}",
            "shape": list(param.shape),
        }

    # add the input shape to the params_dict
    for input_tensor, index in zip(forward_args, range(len(forward_args))):
        inputs_dict[f"input_{index}"] = {
            "shape": list(input_tensor.shape),
        }

    return params_dict, inputs_dict


def strip_main_and_benchmarks(code):
    """
    Strip 'if __name__ == "__main__":' blocks and 'benchmark_compiled_module' functions
    from the given Python code.

    Args:
        code (str): The Python code to process

    Returns:
        str: The processed code with target blocks removed
    """
    # Parse the code into an AST
    tree = ast.parse(code)

    # New list to store the body nodes we want to keep
    new_body = []

    # Filter out the unwanted nodes
    for node in tree.body:
        # Skip "if __name__ == '__main__':" blocks
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and isinstance(node.test.ops[0], ast.Eq)
            and isinstance(node.test.comparators[0], ast.Str)
            and node.test.comparators[0].s == "__main__"
        ):
            continue

        # Skip "benchmark_compiled_module" function definitions
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "benchmark_compiled_module"
        ):
            continue

        # Keep all other nodes
        new_body.append(node)

    # Replace the old body with our filtered body
    tree.body = new_body

    # Convert the modified AST back to source code
    result = astor.to_source(tree)
    return result


def strip_get_input_functions(code):
    """
    Strip get_inputs and get_init_inputs functions from the given Python code.

    Args:

    """
    tree = ast.parse(code)
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in [
            "get_inputs",
            "get_init_inputs",
        ]:
            continue
        new_body.append(node)
    tree.body = new_body
    return astor.to_source(tree)


def swap_forward_call(pytorch_code, entry_point, generated_forward_call):
    # find the forward call and remove it and return the rest of the code
    tree = ast.parse(pytorch_code)

    # Helper function to rename Name nodes
    def rename_references(node):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Name) and child_node.id == entry_point:
                child_node.id = f"{entry_point}New"

    for node in ast.walk(tree):
        # find the module definition for the entry point
        if isinstance(node, ast.ClassDef) and node.name == entry_point:
            # Rename the class definition
            node.name = f"{entry_point}New"

            # Scan all nodes in the class body to replace references to entry_point
            for child in node.body:
                rename_references(child)

                # Additionally, replace the forward method
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    node.body.remove(child)
                    # replace with the generated_forward_call function with proper indentation
                    node.body.append(ast.parse(generated_forward_call).body[0])

    # Also scan for references outside the class definition
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == entry_point:
            node.id = f"{entry_point}New"

    return astor.to_source(tree)


def get_num_outputs_from_forward_call(pytorch_code, entry_point):
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
        if isinstance(child, ast.FunctionDef) and child.name == "forward":
            forward_method = child
            break

    if not forward_method:
        raise ValueError(f"No 'forward' method found in class '{entry_point}'.")

    # Find all return statements in the forward method
    return_nodes = []
    for node in ast.walk(forward_method):
        if isinstance(node, ast.Return):
            return_nodes.append(node)

    # Error if there is more than 1 return statement
    if len(return_nodes) > 1:
        raise ValueError(
            f"Multiple return statements found in the 'forward' method of '{entry_point}'. Only one return statement is allowed."
        )

    if not return_nodes:
        raise ValueError(
            f"No return statement found in the 'forward' method of '{entry_point}'."
        )

    # Get the return value and count outputs
    return_value = return_nodes[0].value

    # Handle different types of return values
    if isinstance(return_value, ast.Tuple):
        # If it's a tuple, count the elements
        return len(return_value.elts)
    else:
        # If it's a single value, return 1
        return 1


def remove_unused_module_level_constants(code):
    """
    Identify and remove unused module-level constants from Python code.
    Returns the cleaned code as a string.

    Args:
        code (str): The Python code as a string

    Returns:
        str: The cleaned code with unused constants removed
    """
    # Parse the code
    tree = ast.parse(code)

    # Find all variable assignments at module level
    constants = {}
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Store variable name and line range
                    constants[target.id] = (
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                    )

    # Find all variable usages
    used_vars = set()

    class VariableVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
            self.generic_visit(node)

    visitor = VariableVisitor()
    visitor.visit(tree)

    # Identify unused constants (defined but not used)
    unused_constants = {
        var: lines for var, lines in constants.items() if var not in used_vars
    }

    # Remove unused constants from the code
    lines = code.split("\n")
    line_indices_to_remove = set()

    for _, (start, end) in unused_constants.items():
        for i in range(start - 1, end):
            line_indices_to_remove.add(i)

    # Filter out lines to remove
    filtered_lines = [
        line for i, line in enumerate(lines) if i not in line_indices_to_remove
    ]

    # Return the cleaned code
    return "\n".join(filtered_lines)


def write_modified_program(pytorch_code, entry_point, triton_code):
    params_dict, input_dict = run_model(pytorch_code, entry_point)
    num_outputs = get_num_outputs_from_forward_call(pytorch_code, entry_point)
    forward_call = write_call_wrapper(triton_code, params_dict, input_dict, num_outputs)
    modified_program = swap_forward_call(pytorch_code, entry_point, forward_call)
    modified_program = strip_get_input_functions(modified_program)
    stripped_triton_code = strip_main_and_benchmarks(triton_code)
    stripped_triton_code = remove_unused_torch_ops(stripped_triton_code)
    combined_code = f"{stripped_triton_code}\n\n{modified_program}"
    linted_combined_code = move_imports_to_top(combined_code)
    linted_combined_code = remove_unused_module_level_constants(linted_combined_code)
    linted_combined_code = run_ruff_on_code(linted_combined_code)
    return linted_combined_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-program", type=str, required=True)
    parser.add_argument("--entry-point", type=str, required=True)
    parser.add_argument("--triton-program", type=str, required=True)
    args = parser.parse_args()

    pytorch_program = args.pytorch_program
    entry_point = args.entry_point
    triton_program = args.triton_program
    pytorch_code = open(pytorch_program, "r").read()
    triton_code = open(triton_program, "r").read()
    output = write_modified_program(pytorch_code, entry_point, triton_code)
    print(output)


if __name__ == "__main__":
    main()
