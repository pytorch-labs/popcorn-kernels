#!/bin/bash
# utils.sh - Common utilities for data processing scripts

# Constants - only declare if not already defined
if [[ -z "${TAIL_LINES+x}" ]]; then
    readonly TAIL_LINES=40000
fi

if [[ -z "${TIMESTAMP_FORMAT+x}" ]]; then
    readonly TIMESTAMP_FORMAT="%Y-%m-%d %H:%M:%S"
fi

if [[ -z "${DEFAULT_TOTAL_CHUNKS+x}" ]]; then
    readonly DEFAULT_TOTAL_CHUNKS=100
fi

if [[ -z "${DEFAULT_CONCURRENT_GPU_JOBS+x}" ]]; then
    readonly DEFAULT_CONCURRENT_GPU_JOBS=8
fi

# Initialize global variables with safe defaults (only if not already set)
: "${JOBS:=1}"
: "${RUN_DIR:=""}"
: "${SHARD_ID:=0}"
: "${NUM_SHARDS:=1}"
: "${SYNTHETIC_DATA_DIR:=""}"
: "${MAIN_LOG_DIR:="logs"}"
: "${TOTAL_CHUNKS:=${DEFAULT_TOTAL_CHUNKS}}"
: "${MASTER_LOG:=""}"
: "${SKIP_DOWNLOAD:=false}"
: "${OUTPUT:=""}"
: "${NUM_CONCURRENT_GPU_JOBS:=${DEFAULT_CONCURRENT_GPU_JOBS}}"

# Create directory structure safely
create_directory_structure() {
    local shard_dir="${MAIN_LOG_DIR}/shard_${SHARD_ID}_of_${NUM_SHARDS}"
    local download_dir="${shard_dir}/download"
    local generate_dir="${shard_dir}/generate"
    local eval_dir="${shard_dir}/evaluate"
    local dataset_dir="${shard_dir}/dataset"

    mkdir -p "${download_dir}" "${generate_dir}" "${eval_dir}" "${dataset_dir}"

    if [[ ! -d "${download_dir}" || ! -d "${generate_dir}" || ! -d "${eval_dir}" || ! -d "${dataset_dir}" ]]; then
        echo "ERROR: Failed to create required directories"
        exit 1
    fi

    # Create a master log file with current datetime if not already defined
    if [[ -z "${MASTER_LOG}" || ! -f "${MASTER_LOG}" ]]; then
        local datetime=$(date +%Y%m%d_%H%M%S)
        MASTER_LOG="${shard_dir}/master_${datetime}.log"
        touch "${MASTER_LOG}"
    fi

    # Also create output directory for backward compatibility
    mkdir -p "output"

    # Create custom output directory if specified
    if [[ -n "${OUTPUT}" ]]; then
        mkdir -p "${OUTPUT}"
        if [[ ! -d "${OUTPUT}" ]]; then
            echo "ERROR: Failed to create output directory: ${OUTPUT}"
            exit 1
        fi
    fi
}

# Function for logging with timestamp to both console and file
log() {
    local log_file="$1"
    local message="$2"
    local timestamp=$(date +"${TIMESTAMP_FORMAT}")
    local log_message="[${timestamp}] ${message}"

    # Print to console with buffer flush
    echo "${log_message}"

    # Write to specified log file with proper redirection
    # Use printf to avoid buffer issues and ensure atomic write
    printf "%s\n" "${log_message}" >> "${log_file}"

    # Also write to master log if defined
    if [[ -n "${MASTER_LOG}" ]]; then
        printf "%s\n" "${log_message}" >> "${MASTER_LOG}"
    fi
}

# Show usage information
show_usage() {
    local log_file="${MAIN_LOG_DIR}/error_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${MAIN_LOG_DIR}"

    log "${log_file}" "Usage: $0 [OPTIONS]"
    log "${log_file}" "Options:"
    log "${log_file}" "  --jobs=NUMBER, -j NUMBER       Number of parallel jobs (required)"
    log "${log_file}" "  --run-dir=PATH, -r PATH        Directory for repos, outputs, and intermediate files (required)"
    log "${log_file}" "  --shard-id=NUMBER, -s NUMBER   Current shard number (0-indexed) (default: 0)"
    log "${log_file}" "  --num-shards=NUMBER, -n NUMBER Total number of shards (default: 1)"
    log "${log_file}" "  --synthetic-data-dir=PATH, -d PATH  Directory with synthetic data (optional)"
    log "${log_file}" "  --total-chunks=NUMBER, -c NUMBER   Number of chunks for generation (default: ${DEFAULT_TOTAL_CHUNKS})"
    log "${log_file}" "  --num-concurrent-gpu-jobs=NUMBER, -g NUMBER   Number of concurrent GPU jobs (default: ${DEFAULT_CONCURRENT_GPU_JOBS})"
    log "${log_file}" "  --limit=NUMBER, -l NUMBER      Limit for processing (default: -1, no limit) (does not affect downloads)"
    log "${log_file}" "  --output=PATH, -o PATH         Custom output directory (optional)"
    log "${log_file}" "  --skip-download, -S            Skip download and generate phases"
    log "${log_file}" "  --help, -h                     Show this help message (only part of run_full_pipline.sh)"
    exit 1
}

# Parse and validate command line arguments
parse_arguments() {
    local datetime=$(date +%Y%m%d_%H%M%S)
    local log_file="${MAIN_LOG_DIR}/error_${datetime}.log"
    mkdir -p "${MAIN_LOG_DIR}"

    # Variables to track required parameters
    local jobs_set=false
    local run_dir_set=false

    # If no arguments provided, show usage
    if [[ "$#" -eq 0 ]]; then
        show_usage
    fi

    # Parse all command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --jobs=*)
                JOBS="${1#*=}"
                jobs_set=true
                # Validate jobs parameter is a number
                if ! [[ "${JOBS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: jobs parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -j)
                JOBS="$2"
                jobs_set=true
                # Validate jobs parameter is a number
                if ! [[ "${JOBS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: jobs parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --run-dir=*)
                RUN_DIR="${1#*=}"
                run_dir_set=true
                shift
                ;;
            -r)
                RUN_DIR="$2"
                run_dir_set=true
                shift 2
                ;;
            --shard-id=*)
                SHARD_ID="${1#*=}"
                # Validate shard_id is a number
                if ! [[ "${SHARD_ID}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: shard-id parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -s)
                SHARD_ID="$2"
                # Validate shard_id is a number
                if ! [[ "${SHARD_ID}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: shard-id parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --num-shards=*)
                NUM_SHARDS="${1#*=}"
                # Validate num_shards is a number
                if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: num-shards parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -n)
                NUM_SHARDS="$2"
                # Validate num_shards is a number
                if ! [[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: num-shards parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --num-concurrent-gpu-jobs=*)
                NUM_CONCURRENT_GPU_JOBS="${1#*=}"
                # Validate num_concurrent_gpu_jobs is a number
                if ! [[ "${NUM_CONCURRENT_GPU_JOBS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: num-concurrent-gpu-jobs parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -g)
                NUM_CONCURRENT_GPU_JOBS="$2"
                # Validate num_concurrent_gpu_jobs is a number
                if ! [[ "${NUM_CONCURRENT_GPU_JOBS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: num-concurrent-gpu-jobs parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --synthetic-data-dir=*)
                SYNTHETIC_DATA_DIR="${1#*=}"
                shift
                ;;
            -d)
                SYNTHETIC_DATA_DIR="$2"
                shift 2
                ;;
            --total-chunks=*)
                TOTAL_CHUNKS="${1#*=}"
                # Validate total_chunks is a number
                if ! [[ "${TOTAL_CHUNKS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: total-chunks parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -c)
                TOTAL_CHUNKS="$2"
                # Validate total_chunks is a number
                if ! [[ "${TOTAL_CHUNKS}" =~ ^[0-9]+$ ]]; then
                    log "${log_file}" "Error: total-chunks parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --limit=*)
                LIMIT="${1#*=}"
                # Validate limit is a number
                if ! [[ "${LIMIT}" =~ ^-?[0-9]+$ ]]; then
                    log "${log_file}" "Error: limit parameter must be a number"
                    exit 1
                fi
                shift
                ;;
            -l)
                LIMIT="$2"
                # Validate limit is a number
                if ! [[ "${LIMIT}" =~ ^-?[0-9]+$ ]]; then
                    log "${log_file}" "Error: limit parameter must be a number"
                    exit 1
                fi
                shift 2
                ;;
            --output=*)
                OUTPUT="${1#*=}"
                shift
                ;;
            -o)
                OUTPUT="$2"
                shift 2
                ;;
            --skip-download|-S)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --help|-h)
                show_usage
                ;;
            *)
                log "${log_file}" "Error: Unknown option: $1"
                show_usage
                ;;
        esac
    done

    # Check if required parameters are provided
    if [[ "$jobs_set" == "false" ]]; then
        log "${log_file}" "Error: Missing required parameter: --jobs / -j"
        show_usage
    fi

    if [[ "$run_dir_set" == "false" ]]; then
        log "${log_file}" "Error: Missing required parameter: --run-dir / -r"
        show_usage
    fi

    # Create run_dir if it doesn't exist
    if [[ ! -d "${RUN_DIR}" ]]; then
        mkdir -p "${RUN_DIR}"

        # Verify creation was successful
        if [[ ! -d "${RUN_DIR}" ]]; then
            log "${log_file}" "Error: Failed to create run_dir: ${RUN_DIR}"
            exit 1
        fi
    fi

    # Validate synthetic_data_dir exists if specified
    if [[ -n "${SYNTHETIC_DATA_DIR}" ]]; then
        if [[ ! -d "${SYNTHETIC_DATA_DIR}" ]]; then
            mkdir -p "${SYNTHETIC_DATA_DIR}"
            if [[ ! -d "${SYNTHETIC_DATA_DIR}" ]]; then
                log "${log_file}" "Error: Failed to create synthetic_data_dir: ${SYNTHETIC_DATA_DIR}"
                exit 1
            fi
            log "${MAIN_LOG_DIR}/notice_${datetime}.log" "Created synthetic_data_dir: ${SYNTHETIC_DATA_DIR}"
        fi
    fi

    # Create the appropriate directory structure after parsing arguments
    create_directory_structure

    # Log the parameters
    log "${MASTER_LOG}" "Script started with args: $*"
    log "${MASTER_LOG}" "Using parameters:"
    log "${MASTER_LOG}" "  JOBS: ${JOBS}"
    log "${MASTER_LOG}" "  RUN_DIR: ${RUN_DIR}"
    log "${MASTER_LOG}" "  SHARD_ID: ${SHARD_ID}"
    log "${MASTER_LOG}" "  NUM_SHARDS: ${NUM_SHARDS}"
    log "${MASTER_LOG}" "  TOTAL_CHUNKS: ${TOTAL_CHUNKS}"
    log "${MASTER_LOG}" "  NUM_CONCURRENT_GPU_JOBS: ${NUM_CONCURRENT_GPU_JOBS}"
    log "${MASTER_LOG}" "  SKIP_DOWNLOAD: ${SKIP_DOWNLOAD}"

    if [[ -n "${SYNTHETIC_DATA_DIR}" ]]; then
        log "${MASTER_LOG}" "  SYNTHETIC_DATA_DIR: ${SYNTHETIC_DATA_DIR}"
    fi

    if [[ "${LIMIT}" ]]; then
        log "${MASTER_LOG}" "  LIMIT: ${LIMIT}"
    fi

    if [[ -n "${OUTPUT}" ]]; then
        log "${MASTER_LOG}" "  OUTPUT: ${OUTPUT}"
    fi
}

# Function to summarize execution times
summarize_execution() {
    local start_time=$1
    shift
    local time_labels=("$@")
    local time_values=()

    # Get all remaining arguments as time values
    for value in "$@"; do
        if [[ "${value}" =~ ^[0-9]+$ ]]; then
            time_values+=("${value}")
        fi
    done

    local total_time=$(($(date +%s) - start_time))

    log "${MASTER_LOG}" "Shard ${SHARD_ID} processing complete!"
    log "${MASTER_LOG}" "Total execution time: ${total_time}s"
    log "${MASTER_LOG}" "Summary of times:"

    local num_times=${#time_values[@]}
    for (( i=0; i<num_times; i++ )); do
        if [[ -n "${time_labels[$i]}" && -n "${time_values[$i]}" ]]; then
            log "${MASTER_LOG}" "- ${time_labels[$i]}: ${time_values[$i]}s"
        fi
    done
}
