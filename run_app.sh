#!/bin/bash

# AMD GPU optimization script for steel manufacturing app
echo "🚀 Starting Steel Manufacturing AI with AMD optimizations..."

# Set environment variables to suppress NNPACK warnings
export NNPACK_DISABLE=1
export PYTORCH_NNPACK_DISABLE=1
export OMP_NUM_THREADS=1

# AMD ROCm optimizations
export HSA_ENABLE_SDMA=0
export CUDA_VISIBLE_DEVICES=0

# PyTorch optimizations for AMD (updated variable name)
export PYTORCH_ALLOC_CONF="max_split_size_mb:128"

# Suppress other common warnings
export PYTHONWARNINGS="ignore"

# Run the Streamlit app with stderr filtering for NNPACK warnings
cd /root/steel_manufacturing
python -m streamlit run app.py --server.port 8502 --server.headless true 2> >(grep -v "NNPACK" >&2)

echo "✅ App started successfully!"
