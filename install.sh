#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
INSTALL_DIR="${ROOT_DIR}/install"

chmod +x install/ds_b580.sh
chmod +x install/ds_macos.sh

echo " Author: SomeZarz"
echo " Select environment to install"
echo "==================================="
echo " 1) Arch Linux (Intel ARC B580)"
echo " 2) MacOS (M1)"
echo " 0) Exit"

read -rp "Enter selection [1-2, 0 to exit]: " choice
case "$choice" in
  1)
    echo "Running Arch Linux installer..."
    "${INSTALL_DIR}/ds_b580_arch.sh"
    ;;
  2)
    echo "Running macOS installer..."
    "${INSTALL_DIR}/ds_macos.sh"
    source fyp_hanyang/bin/activate
    ;;
  0)
    echo "Exiting."
    exit 0
    ;;
  *)
    echo "Invalid selection: $choice"
    exit 1
    ;;
esac