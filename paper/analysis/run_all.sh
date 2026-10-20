#!/bin/bash
# run_all.sh -- every analysis of the paper, in order, after measures/run_measures.sh has written
# paper/data/. Rscript and the R library are taken from the environment (RSCRIPT, R_LIBS_USER).
#
#   01  reading-time table            seconds
#   02  estimates, Natural Stories    minutes (11 strategies in parallel)
#   02  estimates, CLASP              about an hour (ordered beta, 25 fits per strategy and subset)
#   03  CLASP rating scale            minutes
#   04  held-out, Natural Stories     about 20 minutes with 11 cores
#   04  held-out, CLASP               several hours; use analysis.sbatch or NCORES=40
#   05-07 figures and tables          seconds
#   appendix/*                        minutes each on Natural Stories; CLASP scripts up to an hour
#
# Usage: NCORES=11 bash analysis/run_all.sh [naturalstories|clasp|figures|appendix|all]

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(dirname "$HERE")
RSCRIPT=${RSCRIPT:-Rscript}
PYTHON=${PYTHON:-python3}
export NCORES=${NCORES:-11}
mkdir -p "$PAPER/logs"
what=${1:-all}

run() {   # run <log name> <script> [args]
  local log=$1; shift
  echo "$(date +%H:%M) $*"
  "$@" > "$PAPER/logs/$log.log" 2>&1
}

if [ "$what" = naturalstories ] || [ "$what" = all ]; then
  run 01_prepare_naturalstories $RSCRIPT "$HERE/01_prepare_naturalstories.R"
  run 02_estimates_naturalstories $RSCRIPT "$HERE/02_estimates.R" naturalstories
  run 04_heldout_naturalstories $RSCRIPT "$HERE/04_heldout.R" naturalstories
fi
if [ "$what" = clasp ] || [ "$what" = all ]; then
  run 02_estimates_clasp $RSCRIPT "$HERE/02_estimates.R" clasp
  run 03_clasp_rating_scale $RSCRIPT "$HERE/03_clasp_rating_scale.R"
  run 04_heldout_clasp $RSCRIPT "$HERE/04_heldout.R" clasp
fi
if [ "$what" = figures ] || [ "$what" = all ]; then
  run 05_figure_estimates $RSCRIPT "$HERE/05_figure_estimates.R"
  run 06_figure_strategies $RSCRIPT "$HERE/06_figure_strategies.R"
  run 07_tables $PYTHON "$HERE/07_tables.py"
fi
if [ "$what" = appendix ] || [ "$what" = all ]; then
  run appendix_position $RSCRIPT "$HERE/appendix/position.R"
  run appendix_established_count $RSCRIPT "$HERE/appendix/established_count.R"
  run appendix_lead_lag $RSCRIPT "$HERE/appendix/lead_lag.R"
  run appendix_word_level $RSCRIPT "$HERE/appendix/word_level.R"
  for dataset in naturalstories clasp; do
    run appendix_linking_cost_$dataset $RSCRIPT "$HERE/appendix/linking_cost.R" $dataset
    run appendix_topic_$dataset $RSCRIPT "$HERE/appendix/topic.R" $dataset
  done
  run appendix_next_mention $RSCRIPT "$HERE/appendix/next_mention.R"
  for dataset in naturalstories clasp; do
    run appendix_referring_forms_$dataset $RSCRIPT "$HERE/appendix/referring_forms.R" $dataset
  done
  run appendix_clasp_surprisal_size $RSCRIPT "$HERE/appendix/clasp_surprisal_size.R"
  run appendix_clasp_language_control $RSCRIPT "$HERE/appendix/clasp_language_control.R"
fi
echo "$(date +%H:%M) done"
