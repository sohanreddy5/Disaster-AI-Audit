#!/usr/bin/env bash
# ==============================================================================
# Master Reproduction Script for Disaster AI Out-of-Distribution Audit Paper
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "DISASTER AI AUDIT: END-TO-END REPRODUCTION PIPELINE"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================================================"

# 1. Environment Verification
if [ ! -d ".venv" ]; then
    echo "[!] Virtual environment not found. Creating with uv..."
    uv venv .venv --python python3.11
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# 2. Run Test Suite
echo ""
echo "[Step 1/8] Running Verification Test Suites..."
python tests/test_splits.py
python tests/test_full_suite.py

# 3. Multi-Split Benchmarking
echo ""
echo "[Step 2/8] Running Multi-Split Benchmarks (Random vs LOEO vs LODTO vs LORO)..."
python scripts/run_benchmark.py --dataset-name crisismmd
python scripts/run_benchmark.py --dataset-name humaid

# 4. Near-Duplicate Leakage Audit
echo ""
echo "[Step 3/8] Running Near-Duplicate Train/Test Leakage Experiment..."
python scripts/run_leakage_impact.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd
python scripts/run_leakage_impact.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid

# 5. Domain Generalization Benchmark (26 Folds)
echo ""
echo "[Step 4/8] Running Domain Generalization Benchmark (ERM vs GroupDRO vs IRM vs CORAL)..."
python scripts/run_domain_generalization.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd --epochs 15
python scripts/run_domain_generalization.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid --epochs 8 --batch-size 512

# 6. Compound Shift Benchmark
echo ""
echo "[Step 5/8] Running Compound Shift (Type x Region + Sensor Stress)..."
python scripts/run_compound_shift.py --features-npz cache/features/crisismmd_humanitarian_text_features.npz --dataset-name crisismmd
python scripts/run_compound_shift.py --features-npz cache/features/humaid_benchmark_features.npz --dataset-name humaid

# 7. Statistical Power Analysis & Significance Tests
echo ""
echo "[Step 6/8] Running Statistical Power Analysis and Significance Tests..."
python scripts/run_power_analysis.py --features-npz cache/features/humaid_benchmark_features.npz
python scripts/compute_significance_tests.py

# 8. Generate Figures & Export Manifests
echo ""
echo "[Step 7/8] Generating Camera-Ready Publication Figures (PDF & PNG)..."
python scripts/generate_paper_figures.py

echo ""
echo "[Step 8/8] Exporting Split Manifests for Zenodo Release..."
python scripts/export_benchmark_manifest.py

echo ""
echo "========================================================================"
echo "[SUCCESS] All experiments, audits, tables, and figures reproduced!"
echo "Outputs stored in: ${PROJECT_ROOT}/outputs/"
echo "LaTeX Paper in:     ${PROJECT_ROOT}/docs/paper/main.tex"
echo "========================================================================"
