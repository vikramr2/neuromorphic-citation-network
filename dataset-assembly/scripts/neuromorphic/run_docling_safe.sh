#!/bin/bash

# Safe wrapper script that restarts docling_fetch.py if it crashes
# This handles segfaults and other fatal errors

cd "$(dirname "$0")"

# Activate virtual environment
source /home/vr9/vikram_venv/bin/activate

MAX_RETRIES=5
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "=========================================="
    echo "Starting docling_fetch.py (attempt $((retry_count + 1))/$MAX_RETRIES)"
    echo "=========================================="

    python docling_fetch.py
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Processing completed successfully!"
        exit 0
    elif [ $exit_code -eq 139 ]; then
        echo "Segmentation fault detected! Restarting..."
        retry_count=$((retry_count + 1))
        sleep 2
    else
        echo "Process exited with code $exit_code. Restarting..."
        retry_count=$((retry_count + 1))
        sleep 2
    fi
done

echo "Max retries reached. Check the output for errors."
exit 1
