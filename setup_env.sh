#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Super Resolution & PCB Component Counting ==="
echo ""

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install --upgrade pip

echo ""
echo "--- PyTorch installation ---"
echo "Choose one:"
echo "  1) CPU only"
echo "  2) CUDA 12.8"
echo "  3) CUDA 12.4 (for older GPUs like GTX 1060 Pascal)"
echo "  4) Skip (PyTorch already installed)"
read -rp "Select [1-4]: " choice

case "$choice" in
    1)
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
    2)
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        ;;
    3)
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        ;;
    4)
        echo "Skipping PyTorch."
        ;;
    *)
        echo "Invalid choice. Install PyTorch manually."
        ;;
esac

echo ""
echo "Installing application dependencies..."
python3 -m pip install -r requirements.txt

echo ""
echo "Done. Run: ./srpcb.sh"
