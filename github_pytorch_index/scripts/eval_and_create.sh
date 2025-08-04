#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# evaluate_and_create.sh - Evaluation and dataset creation pipeline

# Source the utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Function to run the evaluation step
run_evaluation() {
    local shard_dir="${MAIN_LOG_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}"
    local eval_dir="${shard_dir}/evaluate"
    local datetime=$(date +%Y%m%d_%H%M%S)
    local eval_log="${eval_dir}/evaluate_${datetime}.log"
    local eval_output="${eval_dir}/output_${datetime}.txt"
    local output_legacy="output/evaluate_output_shard${SHARD_ID}.txt"

    log "${MASTER_LOG}" "Starting data evaluation..."
    local eval_start=$(date +%s)

    # Create a temporary file for output
    local temp_output=$(mktemp)

    # Build the evaluate command safely
    local evaluate_cmd="TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python main.py --evaluate-all --compile_mode dynamo --backend inductor --device cuda --jobs ${NUM_CONCURRENT_GPU_JOBS} --shard_num ${SHARD_ID} --shard_total ${NUM_SHARDS} --run-dir \"${RUN_DIR}\""

    # Add synthetic_data_dir if provided
    if [[ -n "${SYNTHETIC_DATA_DIR}" ]]; then
        log "${MASTER_LOG}" "Using synthetic data directory: ${SYNTHETIC_DATA_DIR}"
        evaluate_cmd="${evaluate_cmd} --synthetic-data-dir \"${SYNTHETIC_DATA_DIR}\""
    fi

    # Add limit
    if [[ -n "${LIMIT}" ]]; then
        log "${MASTER_LOG}" "Using limit ${LIMIT}"
        evaluate_cmd="${evaluate_cmd} --limit \"${LIMIT}\""
    fi

    log "${eval_log}" "Executing: ${evaluate_cmd}"

    # Execute with proper redirection
    eval "${evaluate_cmd} > >(tee -a \"${temp_output}\") 2> >(tee -a \"${temp_output}\" >&2)"
    local status=$?

    if [[ ${status} -ne 0 ]]; then
        log "${eval_log}" "WARNING: Evaluation command exited with status ${status}"
    fi

    # Safely copy the output
    tail -n "${TAIL_LINES}" "${temp_output}" > "${eval_output}"
    tail -n "${TAIL_LINES}" "${temp_output}" > "${output_legacy}"
    cat "${temp_output}" >> "${eval_log}"
    rm "${temp_output}"

    local eval_time=$(($(date +%s) - eval_start))
    log "${MASTER_LOG}" "Data evaluation completed in ${eval_time}s"

    return ${status}
}

# Function to create the dataset
create_dataset() {
    local shard_dir="${MAIN_LOG_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}"
    local dataset_dir="${shard_dir}/dataset"
    local datetime=$(date +%Y%m%d_%H%M%S)
    local dataset_log="${dataset_dir}/dataset_${datetime}.log"
    local dataset_output="${dataset_dir}/output_${datetime}.txt"
    local output_legacy="output/create_dataset_output_shard${SHARD_ID}.txt"

    log "${MASTER_LOG}" "Starting dataset creation..."
    local dataset_start=$(date +%s)

    # Create a temporary file for output
    local temp_output=$(mktemp)

    # Build the dataset command safely
    local dataset_cmd="python create_dataset.py --jobs ${JOBS} --run-dir \"${RUN_DIR}\"" --num-concurrent-gpu-jobs ${NUM_CONCURRENT_GPU_JOBS}

    # Add limit
    if [[ -n "${LIMIT}" ]]; then
        log "${MASTER_LOG}" "Using limit ${LIMIT}"
        dataset_cmd="${dataset_cmd} --limit \"${LIMIT}\""
    fi

    # Add output directory
    if [[ -n "${OUTPUT}" ]]; then
        log "${MASTER_LOG}" "Using output directory ${OUTPUT}"
        dataset_cmd="${dataset_cmd} --output-file \"${OUTPUT}\""
    fi

    log "${dataset_log}" "Executing: ${dataset_cmd}"

    # Execute with proper redirection
    eval "${dataset_cmd} > >(tee -a \"${temp_output}\") 2> >(tee -a \"${temp_output}\" >&2)"
    local status=$?

    if [[ ${status} -ne 0 ]]; then
        log "${dataset_log}" "WARNING: Dataset creation exited with status ${status}"
    fi

    # Safely copy the output
    tail -n "${TAIL_LINES}" "${temp_output}" > "${dataset_output}"
    tail -n "${TAIL_LINES}" "${temp_output}" > "${output_legacy}"
    cat "${temp_output}" >> "${dataset_log}"
    rm "${temp_output}"

    local dataset_time=$(($(date +%s) - dataset_start))
    log "${MASTER_LOG}" "Dataset creation completed in ${dataset_time}s"

    return ${status}
}

# Main function to orchestrate the evaluation workflow
main() {
    # Parse and validate input arguments
    parse_arguments "$@"

    # Record overall start time
    local start_time=$(date +%s)

    # Run evaluation
    local eval_start=$(date +%s)
    run_evaluation
    local eval_status=$?
    local eval_time=$(($(date +%s) - eval_start))

    # Create dataset if evaluation was successful
    local dataset_time=0
    if [[ ${eval_status} -eq 0 ]]; then
        local dataset_start=$(date +%s)
        create_dataset
        dataset_time=$(($(date +%s) - dataset_start))
    else
        log "${MASTER_LOG}" "WARNING: Evaluation had issues but attempting dataset creation anyway"
        local dataset_start=$(date +%s)
        create_dataset
        dataset_time=$(($(date +%s) - dataset_start))
    fi

    # Summarize execution
    summarize_execution "${start_time}" "Calculation" "${calc_time}" "Evaluation" "${eval_time}" "Dataset creation" "${dataset_time}"

    return 0
}

# If this script is being executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Run the main function with all arguments
    main "$@"
fi
