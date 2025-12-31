#!/bin/bash
#
# llamacpp_batch.sh - Batch test wrapper for llama.cpp
#
# Since llama.cpp loads models at server startup, this script manages
# the container lifecycle to test multiple models sequentially.
#
# Usage:
#   ./llamacpp_batch.sh <config_file> <results_dir> [options]
#
# Examples:
#   ./llamacpp_batch.sh models.txt results/llamacpp/
#   ./llamacpp_batch.sh models.txt results/ --tensor-split "0.5,0.5"
#   ./llamacpp_batch.sh models.txt results/ --gpu 1660super-2x
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
CONTAINER_NAME="llamacpp"
GGUF_DIR="/mnt/llm-testing/models/gguf"
PORT=8080
TENSOR_SPLIT=""
GPU_NAME="1660super-1x"
COOLDOWN=30
N_GPU_LAYERS=999
CONTEXT_SIZE=4096

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $0 <config_file> <results_dir> [options]

Arguments:
    config_file     File with model paths (one per line)
    results_dir     Directory to store results

Options:
    --container NAME    Container name (default: llamacpp)
    --gguf-dir PATH     GGUF models directory (default: /mnt/llm-testing/models/gguf)
    --port PORT         Server port (default: 8080)
    --tensor-split TS   Tensor split ratios (e.g., "0.5,0.5" for 2 GPUs)
    --gpu NAME          GPU config name (default: 1660super-1x)
    --cooldown SEC      Cooldown between models (default: 30)
    --n-gpu-layers N    GPU layers to offload (default: 999)
    --context-size N    Context size (default: 4096)
    --dry-run           Show what would be done without executing
    -h, --help          Show this help message

Examples:
    # Single GPU
    $0 models.txt results/llamacpp/

    # Dual GPU with 50/50 split
    $0 models.txt results/llamacpp/ --tensor-split "0.5,0.5" --gpu 1660super-2x

    # Mixed 3090 Ti + 3060 Ti (75/25 split)
    $0 models.txt results/llamacpp/ --tensor-split "0.75,0.25" --gpu 3090ti-3060ti
EOF
    exit 1
}

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
CONFIG_FILE=""
RESULTS_DIR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --gguf-dir)
            GGUF_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-split)
            TENSOR_SPLIT="$2"
            shift 2
            ;;
        --gpu)
            GPU_NAME="$2"
            shift 2
            ;;
        --cooldown)
            COOLDOWN="$2"
            shift 2
            ;;
        --n-gpu-layers)
            N_GPU_LAYERS="$2"
            shift 2
            ;;
        --context-size)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$CONFIG_FILE" ]]; then
                CONFIG_FILE="$1"
            elif [[ -z "$RESULTS_DIR" ]]; then
                RESULTS_DIR="$1"
            else
                log_error "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_FILE" ]] || [[ -z "$RESULTS_DIR" ]]; then
    log_error "Missing required arguments"
    usage
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print configuration
echo "========================================"
echo "LLAMA.CPP BATCH TEST"
echo "========================================"
echo "Config file:    $CONFIG_FILE"
echo "Results dir:    $RESULTS_DIR"
echo "GGUF dir:       $GGUF_DIR"
echo "Container:      $CONTAINER_NAME"
echo "Port:           $PORT"
echo "GPU:            $GPU_NAME"
echo "Tensor split:   ${TENSOR_SPLIT:-auto}"
echo "GPU layers:     $N_GPU_LAYERS"
echo "Context size:   $CONTEXT_SIZE"
echo "Cooldown:       ${COOLDOWN}s"
echo "========================================"

if $DRY_RUN; then
    log_warn "DRY RUN - no commands will be executed"
fi

# Count models
MODEL_COUNT=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | wc -l)
log_info "Found $MODEL_COUNT models to test"

# Stop existing container
stop_container() {
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        log_info "Stopping container: $CONTAINER_NAME"
        if ! $DRY_RUN; then
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
        fi
    fi
}

# Start container with model
start_container() {
    local model_path="$1"

    log_info "Starting container with model: $model_path"

    # Build docker command
    local CMD="docker run -d --gpus all --name $CONTAINER_NAME"
    CMD+=" -v $GGUF_DIR:/models"
    CMD+=" -p $PORT:8080"
    CMD+=" ghcr.io/ggerganov/llama.cpp:full-cuda"
    CMD+=" --server -m $model_path --host 0.0.0.0"
    CMD+=" --n-gpu-layers $N_GPU_LAYERS"
    CMD+=" --ctx-size $CONTEXT_SIZE"

    # Add tensor split for multi-GPU
    if [[ -n "$TENSOR_SPLIT" ]]; then
        CMD+=" --tensor-split $TENSOR_SPLIT"
    fi

    echo "Command: $CMD"

    if ! $DRY_RUN; then
        eval "$CMD"
    fi
}

# Wait for server health
wait_for_health() {
    local timeout=120
    local interval=5
    local elapsed=0

    log_info "Waiting for llama.cpp to load model..."

    while [[ $elapsed -lt $timeout ]]; do
        if $DRY_RUN; then
            log_info "[DRY RUN] Would wait for health check"
            return 0
        fi

        if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"status"'; then
            status=$(curl -s "http://localhost:$PORT/health" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            if [[ "$status" == "ok" ]] || [[ "$status" == "ready" ]]; then
                log_info "Model loaded successfully"
                return 0
            fi
            echo -n "."
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Health check timeout after ${timeout}s"
    return 1
}

# Run Python test script
run_tests() {
    local model_name="$1"

    log_info "Running tests for: $model_name"

    if $DRY_RUN; then
        log_info "[DRY RUN] Would run tests"
        return 0
    fi

    python3 "$SCRIPT_DIR/run_test.py" \
        --engine llamacpp \
        --model "$model_name" \
        --gpu "$GPU_NAME" \
        --url "http://localhost:$PORT" \
        --results-dir "$RESULTS_DIR"
}

# Main loop
CURRENT=0
SUCCESSFUL=0
FAILED=0

while IFS= read -r model || [[ -n "$model" ]]; do
    # Skip comments and empty lines
    [[ "$model" =~ ^#.*$ || -z "$model" ]] && continue

    CURRENT=$((CURRENT + 1))

    echo ""
    echo "========================================"
    echo "MODEL $CURRENT/$MODEL_COUNT: $model"
    echo "========================================"

    # Extract model name from path for results
    MODEL_NAME=$(basename "$model" .gguf)

    # Stop existing container
    stop_container

    # Start new container with model
    if ! start_container "$model"; then
        log_error "Failed to start container"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Wait for model to load
    if ! wait_for_health; then
        log_error "Model failed to load"
        stop_container
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run tests
    if run_tests "$MODEL_NAME"; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        log_warn "Tests completed with errors"
        FAILED=$((FAILED + 1))
    fi

    # Stop container
    stop_container

    # Cooldown
    if [[ $CURRENT -lt $MODEL_COUNT ]]; then
        log_info "Cooling down for ${COOLDOWN}s..."
        if ! $DRY_RUN; then
            sleep "$COOLDOWN"
        fi
    fi

done < "$CONFIG_FILE"

# Print summary
echo ""
echo "========================================"
echo "BATCH COMPLETE"
echo "========================================"
echo "Total models:  $MODEL_COUNT"
echo "Successful:    $SUCCESSFUL"
echo "Failed:        $FAILED"
echo "========================================"

# Exit with error if any failures
[[ $FAILED -gt 0 ]] && exit 1
exit 0
