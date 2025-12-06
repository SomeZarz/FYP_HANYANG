#!/usr/bin/env bash

set -euo pipefail

echo "MacOS Environment Setup"
if [[ "$OSTYPE" != "darwin"* ]]; then
  echo "Error: This script must be run on macOS" >&2
  exit 1
fi

echo -e "\nHomebrew Installation & Update"
if command -v brew &> /dev/null; then
  echo "Homebrew found. Updating..."
  brew update
  echo "Homebrew updated successfully"
else
  echo "Homebrew missing. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  echo "Homebrew installed successfully"
fi

echo -e "\nPython 3.11 Installation"
if command -v python3.11 &> /dev/null; then
  echo "Python 3.11 found: $(python3.11 --version)"
else
  echo "Python 3.11 missing. Installing with Homebrew..."
  brew install python@3.11
  echo "Python 3.11 installed: $(python3.11 --version)"
fi

echo -e "\nCreating VE [fyp_hanyang]"
if [[ -d "fyp_hanyang" ]]; then
  echo "Virtual environment 'fyp_hanyang' already exists. Skipping..."
else
  echo "Creating virtual environment 'fyp_hanyang' with python3.11..."
  python3.11 -m venv fyp_hanyang
  echo "Virtual environment 'fyp_hanyang' created."
fi

echo -e "\nActivating VE & PIP install"
source "fyp_hanyang/bin/activate"
python -m pip
# Core PyTorch + vision/audio
pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1
# Data / ML stack
pip install pandas==2.3.3 numpy==1.26.4 scipy==1.16.3 scikit-learn==1.7.2 tqdm==4.67.1 pyyaml==6.0.3 h5py==3.15.1
# Visualization
pip install matplotlib==3.10.7 seaborn==0.13.2 memory-profiler==0.61.0

echo "=================================="
echo -e "\nEnvironment ready: fyp_hanyang"