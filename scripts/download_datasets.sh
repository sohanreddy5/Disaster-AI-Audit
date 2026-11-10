#!/usr/bin/env bash
# ==============================================================================
# Dataset Download and Setup Guide for Disaster AI Audit Benchmark
# ==============================================================================
# This script downloads or provides reproducible fetch commands for all 6 datasets:
#   1. CrisisMMD v2.0 (multimodal social media - 7 events)
#   2. HumAID (text NLP - 19 events)
#   3. MEDIC (social disaster images - ~71k images)
#   4. xBD / xView2 (satellite building damage - 19 events)
#   5. BRIGHT (VHR optical + SAR satellite damage)
#   6. DisasterScope (multimodal event bundles - 24 events)
# ==============================================================================

set -euo pipefail

DATA_ROOT="${1:-./data}"
mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "======================================================================="
echo "Disaster AI Audit: Dataset Setup at ${DATA_ROOT}"
echo "======================================================================="

# --- 1. CrisisMMD v2.0 ---
echo "[1/6] Setting up CrisisMMD v2.0..."
mkdir -p crisismmd
if [ ! -f crisismmd/CrisisMMD_v2.0.tar.gz ] && [ ! -d crisismmd/CrisisMMD_v2.0 ]; then
    echo "Downloading CrisisMMD v2.0 tarball (~2 GB)..."
    curl -L "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz" -o crisismmd/CrisisMMD_v2.0.tar.gz || echo "Direct curl failed; use HuggingFace: datasets.load_dataset('QCRI/CrisisMMD')"
    if [ -f crisismmd/CrisisMMD_v2.0.tar.gz ]; then
        tar -xzf crisismmd/CrisisMMD_v2.0.tar.gz -C crisismmd/
    fi
else
    echo "CrisisMMD archive already exists."
fi

# --- 2. HumAID ---
echo "[2/6] Setting up HumAID (~77k tweets, 19 events)..."
echo "HumAID can be loaded directly in Python via Hugging Face Hub:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('QCRI/HumAID-all')"
mkdir -p humaid

# --- 3. MEDIC ---
echo "[3/6] Setting up MEDIC (~71k images)..."
mkdir -p medic
if [ ! -f medic/MEDIC.tar.gz ] && [ ! -d medic/MEDIC ]; then
    echo "Direct archive URL: https://crisisnlp.qcri.org/data/medic/MEDIC.tar.gz"
    echo "Or Hugging Face: datasets.load_dataset('QCRI/MEDIC')"
fi

# --- 4. xBD / xView2 ---
echo "[4/6] Setting up xBD / xView2 (Satellite Building Damage, 19 events)..."
mkdir -p xbd
echo "xBD requires free registration at https://xview2.org/dataset (CC BY-NC-SA 4.0)."
echo "Download challenge tarballs (train, test, hold) into ${DATA_ROOT}/xbd/"

# --- 5. BRIGHT ---
echo "[5/6] Setting up BRIGHT (Optical + SAR)..."
mkdir -p bright
echo "BRIGHT can be fetched from Zenodo or Hugging Face Hub:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('Kullervo/BRIGHT')"

# --- 6. DisasterScope ---
echo "[6/6] Setting up DisasterScope (24 events, 12,159 bundles)..."
mkdir -p disasterscope
echo "DisasterScope repository: https://github.com/Jieli22/DisasterScope"
echo "Download bundle JSONs and imagery to ${DATA_ROOT}/disasterscope/"

echo "======================================================================="
echo "Setup complete. Run 'python scripts/extract_features.py --dataset crisismmd' next."
echo "======================================================================="
