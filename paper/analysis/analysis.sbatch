#!/bin/bash
#SBATCH --job-name=paper_analysis
#SBATCH --partition=genoa.p
#SBATCH --cpus-per-task=40
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/analysis_%j.log
# Cluster driver for analysis/run_all.sh on the analysis machine. Submit from paper/, e.g.
#   sbatch analysis/analysis.sbatch clasp
export RSCRIPT=/hits/basement/nlp/banerjsk/conda-envs/entiv/bin/Rscript
export R_LIBS_USER=/hits/fast/nlp/banerjsk/dual_mode_coherence_audit/rlib
export PYTHON=/hits/fast/nlp/banerjsk/dual_mode_coherence_audit/.venv/bin/python
export NCORES=40
bash analysis/run_all.sh "${1:-all}"
