#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh # Please replace this to your conda's path

ENV_NAME="environment2"
PYTHON_VERSION="3.9"

echo "Creating Conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

echo "Installing ipykernel"
conda install ipykernel -y
echo "Installation complete."

echo "Installing pytorch and pygs"
pip install "whls/torch-2.0.0+cu117-cp39-cp39-linux_x86_64.whl"
pip install "whls/pyg_lib-0.4.0+pt20cu117-cp39-cp39-linux_x86_64.whl"
pip install "whls/torch_cluster-1.6.3+pt20cu117-cp39-cp39-linux_x86_64.whl"
pip install "whls/torch_sparse-0.6.18+pt20cu117-cp39-cp39-linux_x86_64.whl"
pip install "whls/torch_scatter-2.1.2+pt20cu117-cp39-cp39-linux_x86_64.whl"
pip install "whls/torch_spline_conv-1.2.2+pt20cu117-cp39-cp39-linux_x86_64.whl"
echo "Installation complete."

echo "Installing torchdrug"
pip install torchdrug
echo "Installation complete."

echo "Installing sklearn"
conda install scikit-learn -y
echo "Installation complete."

echo "Installing pandas"
conda install pandas -y
conda install openpyxl -y
echo "Installation complete."

echo "Environment $ENV_NAME has been successfully created and activated."

conda env list