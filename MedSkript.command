#!/bin/bash
# MedSkript.command — macOS double-click launcher
# The terminal window stays open on errors so you can read the output.

# Detect the directory this script lives in (works from any location).
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

# Add Homebrew paths so weasyprint / pango can find their shared libraries.
# Note: do NOT set DYLD_LIBRARY_PATH — causes a Bus Error on Apple Silicon.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

# Prevent deadlocks caused by Docling / PyTorch forking with multiple threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

# Check that the virtual environment exists before trying to launch.
if [ ! -f "$PYTHON" ]; then
    echo ""
    echo "❌ Error: Virtual environment not found."
    echo ""
    echo "Run these commands in the terminal first:"
    echo "  cd $PROJECT_DIR"
    echo "  python3.11 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

echo "🚀 Starting MedSkript..."
echo ""

"$PYTHON" main.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ MedSkript exited with error code $EXIT_CODE."
    echo ""
    read -p "Press Enter to close..."
fi
