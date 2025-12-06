#!/usr/bin/env bash

set -euo pipefail

echo "Arch Linux (Intel Arc B580) Environment Setup"
if ! command -v pacman >/dev/null 2>&1; then
  echo "Error: pacman not found. This script is intended for Arch Linux." >&2
  exit 1
fi
arch="$(uname -m)"
if [[ "${arch}" != "x86_64" ]]; then
  echo "Error: Intel Arc B580 requires x86_64 userspace; detected '${arch}'." >&2
  exit 1
fi

echo -e "\nInstalling Intel GPU runtime and tools..."
# Core OpenCL / Level Zero runtime for Intel GPUs (oneAPI stack)
sudo pacman -S --needed --noconfirm \
  vulkan-intel \
  intel-compute-runtime \
  level-zero-loader \
  level-zero-headers

echo -e "\nInstalling pyenv"
sudo pacman -S --needed --noconfirm pyenv
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"
if ! command -v pyenv >/dev/null 2>&1; then
  echo "Error: pyenv not found on PATH even after installation." >&2
  exit 1
fi

# Install Python 3.11.x if not already installed
PY311_VERSION="${PY311_VERSION:-3.11.14}"
if ! pyenv versions --bare | grep -q "^${PY311_VERSION}\$"; then
  echo -e "\nInstalling Python ${PY311_VERSION} via pyenv (this may take a while)..."
  pyenv install "${PY311_VERSION}"
fi
PY311_BIN="$PYENV_ROOT/versions/${PY311_VERSION}/bin/python"
if [[ ! -x "${PY311_BIN}" ]]; then
  echo "Error: Expected Python 3.11 interpreter not found at ${PY311_BIN}" >&2
  exit 1
fi

ENV_NAME="fyp_hanyang"
echo -e "\nCreating virtual environment [${ENV_NAME}] with Python ${PY311_VERSION}..."

if [[ -d "${ENV_NAME}" ]]; then
  echo "Virtual environment '${ENV_NAME}' already exists. Skipping creation..."
else
  "${PY311_BIN}" -m venv "${ENV_NAME}"
  echo "Virtual environment '${ENV_NAME}' created."
fi

echo -e "\nActivating VE & PIP install"
source "${ENV_NAME}/bin/activate"
# Core PyTorch + vision/audio
pip install "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/xpu
pip install "intel-extension-for-pytorch==2.7.0" -f https://software.intel.com/ipex-whl-stable
# Data / ML stack
pip install pandas==2.3.3 numpy==1.26.4 scipy==1.16.3 scikit-learn==1.7.2 tqdm==4.67.1 pyyaml==6.0.3 h5py==3.15.1
# Visualization
pip install matplotlib==3.10.7 seaborn==0.13.2 memory-profiler==0.61.0

echo "=================================="
echo -e "\nEnvironment ready: ${ENV_NAME}"
