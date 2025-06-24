#!/bin/bash

# top level cache dir
export CACHE_DIR=/matx/u/simonguo/triton_scrape_work_dir/.cache

# Set custom cache directory for PyTorch models
export TORCH_HOME=$CACHE_DIR/torch

# Set custom cache directory for pip packages
export PIP_CACHE_DIR=$CACHE_DIR/pip

# Print confirmation
echo "TORCH_HOME set to: $TORCH_HOME"
echo "PIP_CACHE_DIR set to: $PIP_CACHE_DIR"