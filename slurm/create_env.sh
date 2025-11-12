#!/bin/bash

#==============================================================================
# HPC Module Loading on ETHZ Euler
#==============================================================================
MODULE_STACK="stack/2024-06"
MODULE_GCC="gcc/12.2.0"
MODULE_PYTHON="python_cuda/3.11.6"
MODULE_CUDA="cuda/12.4.1"
MODULE_GITLFS="git-lfs/3.3.0"

echo "Loading HPC modules..."
module load "$MODULE_STACK"
module load "$MODULE_GCC"
module load "$MODULE_PYTHON"
module load "$MODULE_CUDA"
module load "$MODULE_GITLFS"

#==============================================================================
# Create a Python virtual environment
#==============================================================================
VENV_DIR="$HOME/venv/master-thesis"
echo
echo "==== Creating a Python virtual environment in $VENV_DIR ===="

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Virtual environment creation failed."
        exit 1
    fi
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

#==============================================================================
# Set up project directory
#==============================================================================
# Use the directory where this script is located as the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

echo
echo "==== Working in project directory: $PROJECT_DIR ===="
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd into $PROJECT_DIR"; exit 1; }
echo "Current directory: $(pwd)"

#==============================================================================
# Install Dependencies
#==============================================================================
echo
echo "==== Installing requirements from requirements.txt ===="
pip install -r requirements.txt

# Optional: Uncomment if you need PyTorch for ML models
# echo
# echo "==== Installing torch 2.6 for Linux Cuda 12.4 ===="
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

#==============================================================================
# Final Instructions
#==============================================================================
echo
echo "================================================================"
echo " Environment setup complete!"
echo " Virtual environment is located at: $VENV_DIR"
echo " Project directory: $PROJECT_DIR"
echo ""
echo " To re-activate the environment in future sessions, run:"
echo "    source \"$VENV_DIR/bin/activate\""
echo " Then, navigate to the project:"
echo "    cd \"$PROJECT_DIR\""
echo "================================================================"
