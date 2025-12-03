#!/bin/bash
set -e  # Exit on error

echo "Starting Training Pipeline..."
python trainPipeline.py
echo "Training Completed."

echo "Starting Inference Pipeline..."
python inference.py
echo "Inference Completed."

echo "Full pipeline executed successfully!"
