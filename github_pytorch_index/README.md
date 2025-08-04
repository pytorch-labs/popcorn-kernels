# GitHub PyTorch Index

A scalable GitHub scraper and neural network module extractor for Project Popcorn, designed to mine real-world PyTorch code and generate corresponding Triton kernel implementations through torch.compile evaluation.

## Overview

This tool systematically crawls GitHub repositories containing PyTorch code, extracts individual `nn.Module` implementations, and evaluates them using PyTorch's Inductor compiler to generate paired datasets of Python source code and optimized Triton GPU kernels.

Built upon [pytorch-jit-paritybench](https://github.com/jansel/pytorch-jit-paritybench), it extends the original functionality with enhanced parallelization, synthetic data integration, and production-scale data processing capabilities.

## Architecture

### Core Pipeline

1. **Repository Discovery & Download**
   - Crawls GitHub using `torch_repos.json` (pre-ranked by stars) (or up to 2000 of the most popular repositories mentioning torch)
   - Downloads repositories in parallel with configurable sharding
   - Supports resumable downloads and error recovery

2. **Module Extraction & Generation**
   - Scans downloaded repositories for PyTorch neural network modules
   - Extracts individual `nn.Module` classes with associated test inputs
   - Generates standalone test files for each module
   - Handles complex module dependencies and imports

3. **Compilation & Evaluation** 
   - Executes modules in PyTorch eager mode for baseline behavior
   - Compiles modules using torch.compile with Inductor backend
   - Compares eager vs compiled outputs for correctness verification
   - Captures generated Triton kernel code from compilation process

4. **Dataset Assembly**
   - Aggregates successful module evaluations into structured datasets
   - Creates parquet files with PyTorch-Triton code pairs
   - Supports integration with synthetic data sources
   - Maintains comprehensive metadata and error reporting

### Key Components

- **`paritybench/`**: Core evaluation and generation logic
  - `crawler.py`: GitHub repository discovery and download
  - `generate.py`: Module extraction and test generation  
  - `evaluate.py`: torch.compile evaluation and comparison
  - `main.py`: Command-line interface and workflow orchestration

- **`scripts/`**: Production deployment utilities
  - `run_full_pipline.sh`: End-to-end pipeline with robust error handling
  - `eval_and_create.sh`: Evaluation and dataset creation subset
  - `utils.sh`: Shared utilities for logging and job management

- **`create_dataset.py`**: Final dataset assembly and parquet generation

## Usage

### Prerequisites

```bash
# Install dependencies (from repository root)
uv pip install -r ../requirements.txt

# Optional: Set cache directories for large-scale processing
source set_env.sh
```

> **Version Compatibility Note**: To reproduce results from the [KernelBook dataset](https://huggingface.co/datasets/GPUMODE/KernelBook), you must use PyTorch 2.5.0. The requirements.txt specifies PyTorch 2.7.1 for security reasons, but you can downgrade if needed:
> ```bash
> pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### Quick Start

Run the complete pipeline (expects torch_repos.json, see step 1 below):
```bash
./scripts/run_full_pipline.sh --jobs=8 --run-dir=runs/experiment1
```

### Step-by-Step Execution

1. **Generate repository list** (optional, takes hours):
   ```bash
   python stack_crawl.py  # Creates torch_repos.json
   ```

2. **Download repositories**:
   ```bash
   python main.py --download --parallel-download --jobs 40 --run-dir runs/run1
   # With custom repository list and sharding:
   python main.py --download --parallel-download --jobs 40 \
     --repos_file torch_repos.json --run-dir runs/run1 \
     --shard_num 0 --shard_total 4
   ```

3. **Extract and generate test modules**:
   ```bash
   python main.py --generate-all --jobs 40 --run-dir runs/run1
   # With chunked processing:
   python main.py --generate-all --jobs 40 --run-dir runs/run1 \
     --generate_chunk_num 1 --generate_num_chunks 4
   ```

4. **Evaluate with torch.compile**:
   ```bash
   # Create necessary directories
   mkdir -p runs/run1/{inductor_logs,inductor_cache}
   
   # Run evaluation with Inductor backend
   TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python main.py --evaluate-all \
     --compile_mode dynamo --backend inductor --device cuda \
     --jobs 20 --run-dir runs/run1
   ```

5. **Create final dataset**:
   ```bash
   python create_dataset.py --run-dir runs/run1
   ```

### Advanced Usage

#### Synthetic Data Integration
```bash
# Evaluate with synthetic PyTorch programs which are in --synthetic_data_dir 
# These can be created by using the synthetic_torch_modules library
python main.py --evaluate-all --compile_mode dynamo --backend inductor \
  --device cuda --jobs 20 --run-dir runs/run1 \
  --synthetic_data_dir ./synthetic_data_examples

# Use with pipeline scripts 
./scripts/run_full_pipline.sh --jobs=8 --run-dir=runs/combined \
  --synthetic-data-dir=../synthetic_torch_modules/generated_programs/
```

#### Sharded Processing
```bash
# Process shard 2 of 64 total shards
./scripts/run_full_pipline.sh --jobs=16 --run-dir=runs/run1 \
  --shard-id=2 --num-shards=64
```

#### Resumable Workflows
```bash
# Skip download and generation, run evaluation only
./scripts/eval_and_create.sh --jobs=8 --run-dir=existing_run \
  --synthetic-data-dir=./synthetic_data_examples
```

## Output Structure

All outputs are organized under the specified `--run-dir`:

```
runs/run1/
├── downloads/              # Raw GitHub repository ZIP files
├── generated_flattened_modules/  # Extracted PyTorch modules
├── cleaned_pytorch_modules/      # Individual runnable modules  
├── cleaned_pytorch_modules_with_kwargs/  # Modules requiring kwargs
├── synthetic_modules/      # Synthetic data evaluation results
├── inductor_logs/         # torch.compile compilation logs
├── inductor_cache/        # Inductor compilation cache
├── cleaned_triton/        # Generated Triton kernel code
├── linted_triton/         # Human-readable Triton code
├── intermediate_datasets/ # Debug and intermediate data
├── locks/                # FileLock coordination for parallel processing
└── datasets/             # Final parquet datasets
    ├── scrape_dataset.parquet    # GitHub repository data
    ├── synthetic_dataset.parquet # Synthetic program data
    └── dataset.parquet          # Combined dataset
```

## Configuration

### Environment Variables
- `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`: Required for kernel name tracking
- Cache directories can be customized via `set_env.sh`

### Performance Tuning
- `--jobs N`: Parallel processing threads (recommend 8-40 based on system)
- `--shard_num`/`--shard_total`: Distribute processing across machines
- `--generate_chunk_num`/`--generate_num_chunks`: Memory-efficient generation

### Compilation Options
- `--compile_mode`: `dynamo` (default) or `torchscript`  
- `--backend`: `inductor` (default), `eager`, or other dynamo backends
- `--device`: `cuda` (default) or `cpu`

## Error Handling

The system includes comprehensive error handling:
- Automatic retry logic for transient failures
- Detailed logging with timestamps and shard information
- Graceful degradation when individual modules fail
- FileLock coordination prevents race conditions in parallel processing

Error reports and statistics are generated in `errors.csv` and detailed logs.

## Development

### Code Style
- Configured via `ruff.toml` (100 char lines, single quotes, tabs)
- Run `ruff check .` and `ruff format .` for linting

### Testing
Individual components can be tested:
```bash
# Test single repository
python main.py -g path/to/repo.zip --tests-dir test_output/

# Test single generated module
python main.py -e generated_test.py --device cuda
```

## Troubleshooting

### Common Issues

1. **Out of disk space**: Use `source set_env.sh` to redirect caches
2. **CUDA out of memory**: Reduce `--jobs` parameter or use `--device cpu`
3. **Download failures**: Check network connectivity and GitHub rate limits
4. **Module extraction errors**: Review `generated_flattened_modules/` for syntax issues

### Performance Optimization

- Use SSD storage for `inductor_cache/` directory
- Enable GPU for faster torch.compile evaluation
- Increase `--jobs` parameter on high-core machines
- Use sharding (`--shard_num`/`--shard_total`) for distributed processing

## Dependencies

Key dependencies managed via root `requirements.txt`:
- `torch==2.7.1` with CUDA support (use 2.5.0 for KernelBook result reproduction)
- `datasets==3.2.0` for parquet handling
- `pandas==2.2.3` for data manipulation  
- `tqdm==4.67.1` for progress tracking
- `ruff==0.9.6` for code formatting

See repository root `requirements.txt` for complete dependency specifications.