# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os  # To remove the temporary file after processing
import subprocess  # To run external commands (i.e. Ruff)
import tempfile  # To create a temporary file to store the code


def run_ruff_on_code(code: str) -> str:
	"""
	Run Ruff on a given Python code string and return its output.

	This function writes the provided Python code into a temporary file,
	calls Ruff on that file using subprocess, captures the output, and then
	removes the temporary file.

	Parameters:
	    code (str): The Python source code to lint.

	Returns:
	    str: The stdout output from Ruff containing linting messages.
	"""
	# Create a temporary file with a .py suffix so Ruff treats it as Python code.
	with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
		tmp_file.write(code)  # Write the code string to the file.
		tmp_filename = tmp_file.name  # Save the temporary file's name for later use.

	try:
		# Run Ruff on the temporary file.
		# - "ruff" should be installed and available in your system's PATH.
		# - capture_output=True and text=True ensure we capture stdout as a string.
		result = subprocess.run(
			[
				'ruff',
				'check',
				tmp_filename,
				'--fix',
				'--unsafe-fixes',
				'--fix-only',
			],
			capture_output=True,
			text=True,
		)
		output = open(tmp_filename).read()
	finally:
		# Remove the temporary file whether or not Ruff runs successfully.
		os.remove(tmp_filename)

	return output


def lint_code_directory(directory: str) -> None:
	"""
	Apply ruff linter to a directory of code files.

	Args:
	    directory: Path to the directory to lint
	"""
	print(f'Linting code in {directory}')
	subprocess.run([
		'ruff',
		'check',
		directory,
		'--fix',
		'--unsafe-fixes',
		'--fix-only',
	])
