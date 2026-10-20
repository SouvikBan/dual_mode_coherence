#!/bin/bash
#SBATCH --job-name=paper_measures
#SBATCH --partition=genoa-hopper-mli.p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/measures_%j.log
# Cluster driver for run_measures.sh (one GPU for the sentence embeddings). Submit from paper/.
export HF_HOME=/hits/fast/nlp/banerjsk/.hf_home
export PYTHON=/hits/fast/nlp/banerjsk/dual_mode_coherence_audit/.venv/bin/python
export PARALLEL=4 OMP_NUM_THREADS=4
bash measures/run_measures.sh
