
### Github Scaper for Project Popcorn 🍿
This is a github scraper designed for project popcorn which is a fork of https://github.com/jansel/pytorch-jit-paritybench/tree/master

### Step by Step Guide

### Step -1: Install requirements
```
uv pip install -r requirements.txt
```

#### Step 0: Generate "torch_repos.json" [optional]
`torch_repos.json` is a list of repos which contain pytorch code from the stack v1 sorted by stars
It will take a couple of hours to generate, but you just need to run
```
python stack_crawl.py
```

## Running things end to end

### Preprovided Scripts

**NOTE: These scripts expect `torch_repos.json` from step 0 to be in the root of the repo.**

We offer two scripts to run things end to end `scripts/run_full_pipeline.sh` and `eval_and_create.sh`

Running `sh scripts/run_full_pipeline.sh [options]` will run the entire pipeline end to end while running `sh eval_and_create.sh [options]` will just run steps 3 and 4 described below

#### Usage

```bash

# Full pipeline with all stages
./scripts/full_pipeline.sh --jobs=<N> --run-dir=<PATH> [OPTIONS]

# Evaluation and dataset creation only
./scripts/evaluate_and_create.sh --jobs=<N> --run-dir=<PATH> [OPTIONS]
```

Some examples are below
```bash
# Full pipeline with 4 jobs, no sharding
./scripts/full_pipeline.sh --jobs=4 --run-dir=runs/run1

# Full pipeline with synthetic data and skipping downloads
./scripts/full_pipeline.sh --jobs=8 --run-dir=runs/run1 --synthetic-data-dir=./synthetic_data_examples --skip-download

# Using sharding (e.g., shard 2 of 64)
./scripts/full_pipeline.sh --jobs=16 --run-dir=runs/run1 --shard-id=2 --num-shards=64

# Only run evaluation and dataset creation
./scripts/evaluate_and_create.sh --jobs=8 --run-dir=runs/run1

# Evaluation with synthetic data
./scripts/evaluate_and_create.sh --jobs=8 --run-dir=runs/run1 --synthetic-data-dir=./synthetic_data_examples
```


#### Notes on synthetic data generation
1. synthetic_data_dir can be anywhere that's accessible on your machine and will be sharded into equal bits if run with one of the above scripts which do sharding.
2. The files will be copied into the `run_dir` for record keeping.
3. The name of the file should match the name of the model (ie. the name of the file is the entry point)
4. For the data itself, expect a nn.module with a forward and init function. We also require a `get_inputs()` function which returns a list of tensors, and a `get_init_inputs()` function that returns a two element tuple or list. The first element is a list of tensors and the second is a dictionary of kwargs. There are examples in `synthetic_data_examples`.
5. If run manually the synthetic data is injected at step 3 (evaluate) as highlighted in that step.
6. The datasets folder in the run_dir will contain up to 3 parquet files being `scrape_dataset.parquet` (only data scraped from github), `synthetic_dataset.parquet` (only synthetic data), and `dataset.parquet` (all data). If a dataset would be empty, it will simply not be created.

#### Step 1: Download repos
```
# This will scrape 2000 repos (if no --limit is set)
python main.py --download --parallel-download --jobs 40 --run-dir runs/run1

# if you are going off the stack. Feel free to change min_index and max_index as you see fit
python main.py --download --parallel-download --jobs 40 --min_index 0 --max_index 5000 --repos_file torch_repos.json --run-dir runs/run1

```

This will be in --run_dir / downloads (by default) this is runs/run1/downloads)


#### Step 2: Generate tests

Note: This step somehow generates lots of pip and checkpoint caches. If you have a machine with limited home disk; as a temporary workaround, you can `source set_env.sh` to set the cache directory to a larger disk.

This step will go through all the zip files in the download directory and generate tests for each repo.
Each test file will contain extracted pytorch programs [and associated inputs and checks]. Furthermore, it splits these programs into individual modules.

```
python main.py --generate-all --jobs 40 --run-dir runs/run1
```
These files will be in the `{run_dir}/generated_flattened_modules` which is by default `runs/run1/generated_flattened_modules`. It also creates `{run_dir}/cleaned_pytorch_modules` which contains individual pytorch modules as well as `{run_dir}/cleaned_pytorch_modules_with_kwargs` to signify which modules have kwargs for later debugging / analysis.

#### Step 3: Parse pytorch programs out of tests

Create a folder `inductor_logs` and `inductor_cache` to store the logs and cache for `torch.compile` inductor compilation in the run-dir which is by default `runs/run1`.

```
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python main.py --evaluate-all --compile_mode dynamo --backend inductor --device cuda --jobs 20 --run-dir runs/run1 --synthetic_data_dir ./synthetic_data_examples

This commands evaluate everything in the test directory by running it in eager mode, running it in inductor mode, and then comparing the outputs.

This functionality is realized in `evaluate_nn_module` in `paritybench/evaluate.py`, which tests a single nn module by 1) Running it in eager mode, 2) Running it in the specified compilation mode (TorchScript/Dynamo/etc.) 3) Comparing outputs between modes 4) Optionally recording performance metrics.

It also injects synthetic data into the pipeline as well.

This creates the `torch.compile` cache in `inductor_cache` and the logs in `inductor_logs`. It also creates `synthetic_modules` to store the synthetic data it is evaluating / compiling.

#### Step 4: Create the dataset
```
python create_dataset.py --run-dir runs/run1

# the resulting output files can be found in `runs/run1/datasets`
```

The datasets folder in the run_dir will contain up to 3 parquet files being `scrape_dataset.parquet` (only data scraped from github), `synthetic_dataset.parquet` (only synthetic data), and `dataset.parquet` (all data). If a dataset would be empty, it will simply not be created.

This step creates a dataset of `<pytorch code, triton code>` pairs.
A bunch of intermediate outputs are also saved in `{run_dir}` as well.
Currently they are
cleaned_pytorch_modules -- individulized runnable pytorch modules with tests
cleaned_triton - torch inductor output which is cleaned into runnable triton code
linted_triton - the triton code is linted to check to make it more human readable
intermediate_datasets - intermediate datasets which are useful for debugging and looking a failure cases
locks - a directory which is used with FileLock to prevent double writes
---


### Everything below is from ParityBench Readme
ParityBench [repo](https://github.com/jansel/pytorch-jit-paritybench/tree/master).


A test suite to measure TorchScript parity with PyTorch on many `nn.Module`s
crawled from popular GitHub projects.

Run
```
pip install -r requirements.txt
# swap to your preferred version of cuda if neccessary
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
python main.py --download
python main.py --generate-all
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python main.py --evaluate-all --compile_mode dynamo --backend inductor --device cuda --jobs 1
```
python create_dataset.py
```

###  Running ParityBench

- [Install conda] with python>=3.8
and create/activate a [conda environment]

- Install requirements:
```
conda install pip
pip install -r requirements.txt
conda install pytorch torchvision cpuonly -c pytorch-nightly
```

- Run `python main.py`, you should see an output like:
```
TorchScript ParityBench:
          total  passing  score
projects   1172      346  29.5%
tests      8292     4734  57.1%
```
A file `errors.csv` is generated containing the top error messages and example
`generated/*` files to reproduce those errors.

[Install conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
[conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


### Regenerate ParityBench

*WARNING*: this will download 10+ gigabytes of code from crawling github and
take days to complete.  It is likely not necessary for you to do this.
```
python main.py --download
python main.py --generate-all
```

### Download, generate, evaluate
You can limit number of github projects to download for testing and running on a smaller set of github repos
```
python main.py --download --download-dir <folder path> --limit 10
```
You can generate tests for one project folder `-g`. This will extract nn modules from that project and generate a test script `--tests-dir`
```
python main.py -g <folder path> --tests-dir <folder path>
```
You can evaluate one generated test script `-e` and try export the module to onnx `--onnxdir`
```
python main.py -e <test.py file> --onnxdir <folder path>
```
You can evaluate using different compile mode, e.g, `dynamo`(default) or `torchscript`.
```
python main.py -e <test.py file> --compile_mode dynamo
```
You can evaluate using different dynamo backends provided in `torch._dynamo`, please refer `torch._dynamo.list_backends()`.
```
python main.py -e <test.py file> --backend eager
```
You can evaluate using `cuda`(default) or `cpu`.
```
python main.py -e <test.py file> --device cuda
```
