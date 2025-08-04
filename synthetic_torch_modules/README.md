# Synthetic Torch Modules

A language model-driven synthetic PyTorch program generator that creates diverse neural network architectures for training data augmentation in Project Popcorn. This tool generates valid PyTorch `nn.Module` implementations that can be compiled with torch.compile to produce PyTorch-Triton code pairs.

## Overview

This system combines operator sampling strategies with large language model capabilities to generate diverse, syntactically correct PyTorch neural network modules. The generated programs serve as training data augmentation for models learning to understand PyTorch code structure and compilation behavior.

### Key Features

- **LLM-Driven Generation**: Uses configurable language model APIs to create programs from operator specifications
- **Operator Diversity**: Samples from curated lists of core, compound, and supporting PyTorch operators
- **Automatic Validation**: Tests generated programs for syntax correctness and torch.compile compatibility  
- **Quality Filtering**: Removes programs similar to KernelBench level 2 to prevent test set contamination
- **Parallel Processing**: Supports multi-worker generation with real-time yield monitoring
- **Configurable Complexity**: Adjustable parameters for operator count and program sophistication

## Architecture

### Generation Workflow

1. **Operator Sampling**
   - Selects combinations from predefined operator categories
   - Core operators: Essential PyTorch functions (Conv2d, Linear, etc.)
   - Compound operators: Complex multi-step operations
   - Supporting operators: Auxiliary functions (activations, normalization, etc.)

2. **LLM Program Synthesis** 
   - Constructs prompts specifying desired operators and constraints
   - Queries language model APIs (Gemini, OpenAI, local servers)
   - Extracts Python code from LLM responses

3. **Validation Pipeline**
   - Syntax checking and AST parsing
   - PyTorch eager execution validation  
   - torch.compile compatibility verification
   - Input/output shape consistency checks

4. **Quality Control**
   - Filters out programs resembling KernelBench test cases
   - Deduplication based on operator signatures
   - Yield rate monitoring and optimization

### Core Components

- **`operators.py`**: Defines categorized PyTorch operator sets
- **`generate_synth_torch.py`**: Main generation orchestrator with multiprocessing
- **`utils.py`**: Validation, testing, and LLM interface utilities
- **`prompts/`**: Configurable prompt templates for different LLM providers

## Usage

### Prerequisites

```bash
# Install dependencies (from repository root)
uv pip install -r ../requirements.txt

# Set up API credentials
echo "GEMINI_API_KEY=your_key_here" > .env
# or configure other LLM providers
```

> **Version Compatibility Note**: To reproduce results from the [KernelBook dataset](https://huggingface.co/datasets/GPUMODE/KernelBook), you must use PyTorch 2.5.0. The requirements.txt specifies PyTorch 2.7.1 for security reasons, but you can downgrade if needed:
> ```bash
> pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### Quick Start

Generate a single program for debugging:
```bash
python3 generate_synth_torch.py .single_debug
```

Generate multiple programs in parallel:
```bash
python3 generate_synth_torch.py .parallel num_total_samples=1000
```

### Configuration Options

All parameters can be specified via command line:
```bash
python3 generate_synth_torch.py .parallel \
  num_total_samples=5000 \
  num_worker=20 \
  model_name=gemini \
  num_core_ops_range=[1,5] \
  num_compound_ops_range=[0,4] \
  num_supporting_ops_range=[2,10] \
  p_value=0.25 \
  output_dir=./generated_programs
```

### Parameter Guide

#### Operator Configuration
- `num_core_ops_range=[min,max]`: Number of core operators to sample
- `num_compound_ops_range=[min,max]`: Number of compound operators to sample  
- `num_supporting_ops_range=[min,max]`: Number of supporting operators to sample

#### Generation Control
- `p_value` (0.0-1.0): Sampling probability; lower values = more operators per program
- `num_total_samples`: Total programs to generate
- `num_worker`: Parallel worker processes
- `model_name`: LLM provider (`gemini`, `openai`, `local`)

#### Quality Optimization
- `temperature`: LLM sampling temperature for creativity vs consistency
- `max_retries`: Maximum attempts per operator combination
- `validation_timeout`: Timeout for program execution testing

### Advanced Usage

#### High-Diversity Generation
```bash
# Generate complex, diverse programs with lower success rate
python3 generate_synth_torch.py .parallel \
  num_total_samples=10000 \
  num_core_ops_range=[3,8] \
  num_compound_ops_range=[1,6] \
  num_supporting_ops_range=[5,15] \
  p_value=0.15 \
  temperature=0.8
```

#### High-Yield Generation  
```bash
# Generate simpler programs with higher success rate
python3 generate_synth_torch.py .parallel \
  num_total_samples=10000 \
  num_core_ops_range=[1,3] \
  num_compound_ops_range=[0,2] \
  num_supporting_ops_range=[2,6] \
  p_value=0.4 \
  temperature=0.3
```

#### Mixed Strategy Generation
```bash
# Run multiple configurations to maximize diversity
python3 generate_synth_torch.py .parallel num_total_samples=2000 p_value=0.1 &
python3 generate_synth_torch.py .parallel num_total_samples=2000 p_value=0.3 &  
python3 generate_synth_torch.py .parallel num_total_samples=2000 p_value=0.5 &
wait
```

## Program Requirements

Generated PyTorch programs must meet these specifications:

### File Structure
```python
# File: SynthModel_Conv2d_ReLU_MaxPool2d.py
import torch
import torch.nn as nn

class SynthModel_Conv2d_ReLU_MaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        # Module definition
        
    def forward(self, x):
        # Forward pass implementation
        return result

def get_inputs():
    """Return list of input tensors for testing"""
    return [torch.randn(1, 3, 224, 224)]
    
def get_init_inputs():
    """Return (args, kwargs) for module initialization"""  
    return [], {}
```

### Constraints
- **File naming**: Must match the class name exactly
- **Required functions**: `get_inputs()` and `get_init_inputs()` must be present
- **Shape consistency**: Input/output shapes must be mathematically valid
- **Execution compatibility**: Must run in both eager and compiled modes

## Output Structure

Generated programs are organized as:
```
output_dir/
├── SynthModel_Conv2d_ReLU_MaxPool2d.py
├── SynthModel_Linear_Dropout_LayerNorm.py  
├── SynthModel_ConvTranspose2d_BatchNorm_GELU.py
└── ...
```

Each file contains a complete, runnable PyTorch program ready for integration with the main pipeline.

## Integration with Main Pipeline

Synthetic programs integrate seamlessly with the GitHub PyTorch Index:

```bash
# Use synthetic data with main pipeline
cd ../github_pytorch_index/
python main.py --evaluate-all --compile_mode dynamo --backend inductor \
  --device cuda --jobs 20 --run-dir runs/combined \
  --synthetic_data_dir ../synthetic_torch_modules/generated_programs/

# Or with pipeline scripts
./scripts/run_full_pipline.sh --jobs=8 --run-dir=runs/combined \
  --synthetic-data-dir=../synthetic_torch_modules/generated_programs/
```

## Quality Control

### Validation Pipeline
1. **Syntax Validation**: AST parsing and import checking
2. **Execution Testing**: Runs in PyTorch eager mode  
3. **Compilation Testing**: Verifies torch.compile compatibility
4. **Shape Validation**: Ensures input/output tensor compatibility
5. **KernelBench Filtering**: Removes contaminating test cases

### Yield Optimization

Monitor generation statistics in real-time:
- **Yield Rate**: Percentage of generated programs that pass validation
- **Operator Success Rates**: Which operators contribute to failures  
- **Error Categories**: Syntax vs runtime vs compilation errors

Optimize yield by:
- Adjusting `p_value` (higher = simpler programs = higher yield)
- Reducing operator ranges for complex categories
- Tuning LLM `temperature` (lower = more conservative)
- Improving prompt engineering in `prompts/` directory

### Error Analysis

Common failure modes and solutions:
- **Shape mismatches**: Review operator combinations for dimensional compatibility
- **Missing imports**: Ensure all operators are properly imported in generated code
- **torch.compile failures**: Some operator combinations may not be supported by Inductor
- **Syntax errors**: Tune LLM temperature and add validation examples to prompts

## Development

### Code Style
Follow repository standards via `ruff.toml`:
```bash
ruff check .
ruff format .  
```

### Extending Operators
Add new operators in `operators.py`:
```python
core_operators = [
    'nn.Conv2d',
    'nn.Linear',
    'nn.YourNewOperator',  # Add here
    # ...
]
```

### Custom LLM Providers
Implement new providers in `utils.py`:
```python
def generate_custom_llm(prompt, **kwargs):
    # Your LLM interface implementation
    return generated_code
```

### Prompt Engineering
Customize generation prompts in `prompts/`:
- `prompts_sahan.toml`: Default prompt configurations
- `prompts_simon.toml`: Alternative prompt strategies

## Performance Tuning

### Generation Speed
- Increase `num_worker` for more parallel processes
- Use local LLM servers to reduce API latency
- Optimize `max_retries` vs `timeout` balance

### Memory Management
- Monitor process memory usage with many workers
- Adjust batch sizes for validation operations  
- Use process pools with limited lifetime

### Cost Optimization (API-based LLMs)
- Tune `p_value` and operator ranges to maximize yield
- Use local models for development and iteration
- Cache and reuse successful operator combinations

## Troubleshooting

### Common Issues
1. **Low yield rates**: Increase `p_value`, reduce operator complexity, tune prompts
2. **API rate limiting**: Reduce `num_worker`, add delay between requests
3. **torch.compile failures**: Update PyTorch version, check operator compatibility  
4. **Memory issues**: Reduce worker count, add timeout controls

### Debugging Tools
```bash
# Test single operator combination
python3 generate_synth_torch.py .single_debug

# Validate existing program
python -c "from utils import test_synthetic_model; test_synthetic_model('path/to/program.py')"

# Check operator coverage
python -c "from operators import *; print(f'Total operators: {len(core_operators + compound_operators + supporting_operators)}')"
```