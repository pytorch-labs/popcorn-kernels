import logging
import os
import shutil

import torch
import torch._logging

# Define the output directory for the logs
output_dir = "compile_logs"
os.makedirs(output_dir, exist_ok=True)

# Configure the logging settings
log_file_path = os.path.join(output_dir, "compile_output.log")
torch._logging.set_logs(
    output_code=True,  # Enable logging of the output code
)


# Define a simple model for demonstration
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


# Instantiate and compile the model
model = MyModel()
compiled_model = torch.compile(model, backend="inductor")

# Run the compiled model with example input to trigger compilation
input_data = torch.randn(2, 10)
output_data = compiled_model(input_data)

print(f"Compilation logs, including generated code, are saved in: {log_file_path}")
