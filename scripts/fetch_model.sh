#!/usr/bin/env bash
# Download + export + quantize the detector into models/. Usage: scripts/fetch_model.sh [hf-model-id]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --quiet torch transformers onnx onnx-ir onnxruntime
python scripts/fetch_model.py ${1:+--model "$1"}
