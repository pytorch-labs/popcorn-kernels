#!/bin/bash
# full_pipeline.sh - Complete data processing workflow

# Source the utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"
source "${SCRIPT_DIR}/eval_and_create.sh"

# Function to download repositories
download_repositories() {
    local shard_dir="${MAIN_LOG_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}"
    local download_dir="${shard_dir}/download"
    local datetime=$(date +%Y%m%d_%H%M%S)
    local download_log="${download_dir}/download_${datetime}.log"
    local download_output="${download_dir}/output_${datetime}.txt"
    local output_legacy="output/download_output_shard${SHARD_ID}.txt"

    log "${MASTER_LOG}" "Starting repository download..."
    local download_start=$(date +%s)

    # Create a temporary file for output
    local temp_output=$(mktemp)

    # Build download command
    local download_cmd="python main.py --download --parallel-download --jobs ${JOBS} --shard_num ${SHARD_ID} --shard_total ${NUM_SHARDS} --repos_file torch_repos.json --run-dir \"${RUN_DIR}\""

    log "${download_log}" "Executing: ${download_cmd}"

    # Execute with proper redirection
    eval "${download_cmd} > >(tee -a \"${temp_output}\") 2> >(tee -a \"${temp_output}\" >&2)"
    local status=$?

    if [[ ${status} -ne 0 ]]; then
        log "${download_log}" "WARNING: Download command exited with status ${status}"
    fi

    # Safely copy the output
    mkdir -p "output"
    tail -n "${TAIL_LINES}" "${temp_output}" > "${download_output}"
    tail -n "${TAIL_LINES}" "${temp_output}" > "${output_legacy}"
    cat "${temp_output}" >> "${download_log}"
    rm "${temp_output}"

    local download_time=$(($(date +%s) - download_start))
    log "${MASTER_LOG}" "Repository download completed in ${download_time}s"

    return ${status}
}

# Function to generate data in chunks
generate_data() {
    local shard_dir="${MAIN_LOG_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}"
    local generate_dir="${shard_dir}/generate"
    local datetime=$(date +%Y%m%d_%H%M%S)
    local generate_log="${generate_dir}/generate_${datetime}.log"
    local generate_output="${generate_dir}/output_${datetime}.txt"
    local output_legacy="output/generate_output_shard${SHARD_ID}.txt"

    log "${MASTER_LOG}" "Starting data generation in ${TOTAL_CHUNKS} chunks..."
    local generate_start=$(date +%s)

    # Create a temporary file for combined output
    local temp_combined_output=$(mktemp)
    local overall_status=0

    for chunk_num in $(seq 1 ${TOTAL_CHUNKS}); do
        log "${generate_log}" "Running generation chunk ${chunk_num} of ${TOTAL_CHUNKS}"
        local chunk_start=$(date +%s)

        # Create a temporary file for this chunk's output
        local temp_output=$(mktemp)

        # Build generate command for this chunk
        local generate_cmd="python main.py --generate-all --jobs ${JOBS} --run-dir \"${RUN_DIR}\" --generate_chunk_num ${chunk_num} --generate_num_chunks ${TOTAL_CHUNKS}"

        log "${generate_log}" "Executing: ${generate_cmd}"

        # Execute with proper redirection
        eval "${generate_cmd} > >(tee -a \"${temp_output}\") 2> >(tee -a \"${temp_output}\" >&2)"
        local chunk_status=$?

        # If any chunk fails, record it but continue with other chunks
        if [[ ${chunk_status} -ne 0 ]]; then
            log "${generate_log}" "WARNING: Generation chunk ${chunk_num} exited with status ${chunk_status}"
            overall_status=1
        fi

        # Append this chunk's output to the combined output
        cat "${temp_output}" >> "${temp_combined_output}"

        # Clean up this chunk's temporary file
        rm "${temp_output}"

        local chunk_time=$(($(date +%s) - chunk_start))
        log "${generate_log}" "Generation chunk ${chunk_num} completed in ${chunk_time}s"
    done

    # Safely copy the combined output
    mkdir -p "output"
    tail -n "${TAIL_LINES}" "${temp_combined_output}" > "${generate_output}"
    tail -n "${TAIL_LINES}" "${temp_combined_output}" > "${output_legacy}"
    cat "${temp_combined_output}" >> "${generate_log}"
    rm "${temp_combined_output}"

    local generate_time=$(($(date +%s) - generate_start))
    log "${MASTER_LOG}" "All data generation chunks completed in ${generate_time}s"

    return ${overall_status}
}

# Main function to orchestrate the full workflow
main() {
    # Parse and validate input arguments
    parse_arguments "$@"

    # Parse additional options
    parse_additional_options "$@"

    # Record overall start time
    local start_time=$(date +%s)

    # Initialize timing variables
    local download_time=0
    local download_status=0
    local generate_time=0
    local generate_status=0

    if [[ "${SKIP_DOWNLOAD}" == "true" ]]; then
        log "${MASTER_LOG}" "Skipping download and generate steps as requested by --skip-download flag"
    else
        # Download repositories
        local download_start=$(date +%s)
        download_repositories
        download_status=$?
        download_time=$(($(date +%s) - download_start))

        # Exit if download failed
        if [[ ${download_status} -ne 0 ]]; then
            log "${MASTER_LOG}" "ERROR: Repository download failed, exiting"
            summarize_execution "${start_time}" "Calculation" "${calc_time}" "Download" "${download_time}"
            exit 1
        fi

        # Generate data
        local generate_start=$(date +%s)
        generate_data
        generate_status=$?
        generate_time=$(($(date +%s) - generate_start))

        # Continue even if generation had some issues
        if [[ ${generate_status} -ne 0 ]]; then
            log "${MASTER_LOG}" "WARNING: Some data generation chunks had issues, continuing anyway"
        fi
    fi

    # Run evaluation
    local eval_start=$(date +%s)
    run_evaluation
    local eval_status=$?
    local eval_time=$(($(date +%s) - eval_start))

    # Create dataset
    local dataset_time=0
    if [[ ${eval_status} -eq 0 ]]; then
        local dataset_start=$(date +%s)
        create_dataset
        local dataset_status=$?
        dataset_time=$(($(date +%s) - dataset_start))

        if [[ ${dataset_status} -ne 0 ]]; then
            log "${MASTER_LOG}" "WARNING: Dataset creation encountered issues"
        fi
    else
        log "${MASTER_LOG}" "WARNING: Evaluation had issues but attempting dataset creation anyway"
        local dataset_start=$(date +%s)
        create_dataset
        dataset_time=$(($(date +%s) - dataset_start))
    fi

    # Summarize execution
    summarize_execution "${start_time}" "Calculation" "${calc_time}" "Download" "${download_time}" \
                       "Generation" "${generate_time}" "Evaluation" "${eval_time}" "Dataset creation" "${dataset_time}"

    return 0
}

# If this script is being executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Show help message if requested
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    fi

    # Run the main function with all arguments
    main "$@"
fi
