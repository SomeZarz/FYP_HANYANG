#!/usr/bin/env bash

set -euo pipefail

echo "=================================="
echo " Arch Linux Environment Setup"
echo "=================================="

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
  echo "Error: This script must be run on macOS" >&2
  exit 1
fi

echo "----------------------------------"
echo " Checking for Pacman Packages"
echo "----------------------------------"

