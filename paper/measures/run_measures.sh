#!/bin/bash
# run_measures.sh -- compute every per-utterance measure of the paper for both datasets.
#
# Reads the annotated alternatives (one JSON per Natural Stories story and per CLASP context) and
# writes paper/data/measures/<measure>/<dataset>/<document>.csv, one row per target and sampling
# strategy, plus the word-level files of Appendix A. The sentence-embedding steps (part_of_story,
# linking_cost, topic, non_entity) use a GPU when one is visible; non_entity encodes every
# alternative sentence and is slow on CPU (hours), the others take minutes.
#
# Usage: bash run_measures.sh            (set PYTHON, DATA_ROOT, GUM_DIR for another machine)

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

HERE=$(cd "$(dirname "$0")" && pwd)
PAPER=$(dirname "$HERE")
PYTHON=${PYTHON:-python}
DATA_ROOT=${ENTITY_IV_DATA:-/hits/fast/nlp/banerjsk/crac2026-corpipe/corpipe_seeded_smoke/vllm_raw_generation}
GUM_DIR=${GUM_DIR:-/hits/fast/nlp/banerjsk/gum/dep}
ALTERNATIVES=$DATA_ROOT/annotated_alternatives
ZONE_MAP=$PAPER/../annotated/surprisal/pythia-70m-deduped-step1000-full.tsv
WORDS=$DATA_ROOT/naturalstories/naturalstories_RTS/all_stories.tok
OUT=$PAPER/data/measures
PARALLEL=${PARALLEL:-10}        # documents scored at the same time
mkdir -p "$PAPER/data" "$PAPER/logs"

echo "start $(date)"

# models fitted on GUM: salience (topic bits) and next mention (Appendix A); about a minute each
if [ ! -f "$PAPER/data/gum_salience_model.json" ]; then
  $PYTHON "$HERE/train_salience.py" "$GUM_DIR" "$PAPER/data/gum_salience_model.json"
fi
if [ ! -f "$PAPER/data/gum_next_mention_model.json" ]; then
  $PYTHON "$HERE/train_next_mention.py" "$GUM_DIR" "$PAPER/data/gum_next_mention_model.json"
fi
GOLD=$DATA_ROOT/annotated_csv          # manual Natural Stories annotation (information status)

# one measure script over every document of one dataset
score() {
  local script=$1 measure=$2 dataset=$3
  shift 3
  mkdir -p "$OUT/$measure/$dataset"
  ls "$ALTERNATIVES/$dataset"/*.json | xargs -P "$PARALLEL" -I{} sh -c \
    "b=\$(basename {} .json); [ -f \"$OUT/$measure/$dataset/\$b.csv\" ] || \
     $PYTHON \"$HERE/$script\" $* {} \"$OUT/$measure/$dataset/\$b.csv\" > \"$PAPER/logs/${measure}_\$b.log\" 2>&1"
  echo "$measure $dataset done $(date)"
}

for dataset in naturalstories clasp; do
  score established_entities.py established_entities $dataset
  score part_of_story.py part_of_story $dataset
  score linking_cost.py linking_cost $dataset
  score topic.py topic $dataset "$PAPER/data/gum_salience_model.json"
  score non_entity.py non_entity $dataset
  # Appendix A, claims tested and not reported: referring forms, next-mention topic
  score referring_forms.py referring_forms $dataset "$PAPER/data/gum_next_mention_model.json"
  score topic_next_mention.py topic_next_mention $dataset "$PAPER/data/gum_next_mention_model.json" "$GOLD"
done
# first-referent and pronoun models (Natural Stories only)
score next_mention.py next_mention naturalstories "$PAPER/data/gum_next_mention_model.json" "$GOLD"

# word-level files for Appendix A (Natural Stories)
$PYTHON "$HERE/first_mention_zones.py" "$ZONE_MAP" "$ALTERNATIVES/naturalstories" "$PAPER/data/naturalstories_first_mention_zones.csv"
$PYTHON "$HERE/word_zipf.py" "$WORDS" "$PAPER/data/naturalstories_word_zipf.csv"
$PYTHON "$HERE/word_surprisal.py" "$WORDS" "$PAPER/data/naturalstories_word_surprisal.csv"

echo "done $(date)"
