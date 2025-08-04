#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######
# Bash Script to Sweep over different configs for the synthesis generation process
######

TOTAL_ITERATIONS=200
PER_ITERATION_TOTAL_SAMPLES=8000
PER_ITERATION_TOTAL_WORKERS=2000
FAKE_COMMAND=("python3 print_hi.py" "python3 print_hi.py" "python3 print_hi.py" "python3 print_hi.py" "python3 print_hi.py" "python3 print_hi.py" "python3 print_hi.py")

TIMEOUT_SECONDS=600

# You can list a variety of configs to sweep different variations
COMMANDS=(
    # optimzie for more programs 
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS p_value=0.5"


    # most balanced config
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS p_value=0.3"

    # find less common programs
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS p_value=0.2"
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS p_value=0.1"

    # long sequence
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS p_value=0.3 max_tokens=4096"

    # change composition of operators
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS num_core_ops_range=[1,5] num_compound_ops_range=[0,4] num_supporting_ops_range=[2,10] p_value=0.25"
    "python3 generate_synth_torch.py .parallel model_name=local num_total_samples=$PER_ITERATION_TOTAL_SAMPLES num_worker=$PER_ITERATION_TOTAL_WORKERS num_core_ops_range=[2,4] num_compound_ops_range=[0,4] num_supporting_ops_range=[3,9] p_value=0.25"
)

for ((i=1; i<=$TOTAL_ITERATIONS; i++)); do
    echo "Running iteration $i of $TOTAL_ITERATIONS..."
    
    # Select command (cycling through the list)
    CMD_INDEX=$(( (i-1) % ${#COMMANDS[@]} ))
    CURRENT_CMD="${COMMANDS[$CMD_INDEX]}"

    echo "Running command: $CURRENT_CMD"
    
    # Run the selected command with timeout - show all output but save only last 50 lines
    # timeout 1200 $CURRENT_CMD | tee >(tail -n 50 >> "gen_output_toka.txt")
    timeout $TIMEOUT_SECONDS $CURRENT_CMD
    TIMEOUT_EXIT_CODE=$?

    if [ $TIMEOUT_EXIT_CODE -eq 124 ]; then
        echo "Iteration $i timed out after 20 minutes"
    else
        echo "Completed iteration $i"
    fi
        
    # Kill any existing python3 processes before starting new iteration
    # this is to clear any stale process / worker / http connections
    killall python3 2>/dev/null || true

    sleep 10
done

echo "All iterations completed!"
