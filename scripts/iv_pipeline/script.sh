#generate natural stories alternatives
python generate_ns_alternatives.py  --csv-dir annotated_csv/   --output-dir raw_ns    --model ../../entity_IV/google/gemma-2-9b    --gpus 4
python filter_raw_ns.py --raw-dir raw_ns/ --output-dir filtered_ns


#corpipe26_twostage.py should come from the crac2026-corpipe repo
# COMMON_ARGS=(
#   --entity-dir annotated_csv
#   --alternatives-dir filtered_ns
#   --corpipe-source ../corpipe26_twostage.py
#   --out-dir annotated_ns_alternatives
#   --device cuda
#   --stanza-gpu
#   --threads 2
#   --branch-limit 100
#   --branch-batch-size 25
# )

# mkdir -p annotation_logs

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
#   python annotate_ns_alternatives.py "${COMMON_ARGS[@]}" \
#   --stories 1 4 \
#   > annotation_logs/gpu0.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
#   python annotate_ns_alternatives.py "${COMMON_ARGS[@]}" \
#   --stories 2 5 7 \
#   > annotation_logs/gpu1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
#   python annotate_ns_alternatives.py "${COMMON_ARGS[@]}" \
#   --stories 3 8 10 \
#   > annotation_logs/gpu2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
#   python annotate_ns_alternatives.py "${COMMON_ARGS[@]}" \
#   --stories 6 9 \
#   > annotation_logs/gpu3.log 2>&1 &

# wait
# echo "All annotation processes finished."



##CLASP

python generate_clasp_alternatives.py \
   --clasp-data ../../entity_IV/BLL2018/data/processed_ratings.csv \
   --model ../../entity_IV/google/gemma-2-9b \
   --output-dir raw_clasp \
  --gpus 0,1,2,3 \
   --batch-size 8


python filter_raw_clasp.py \
   --raw-dir raw_clasp \
   --output-dir filtered_clasp \
   --n 100


COMMON_ARGS=(
  --clasp-alternatives filtered_clasp
  --clasp-gold ../../entity_IV/BLL2018/data/processed_ratings.csv
  --corpipe-source ../corpipe26_twostage.py
  --out-dir annotated_clasp_alternatives
  --device cuda
  --stanza-gpu
  --branch-limit 100
)

# mkdir -p annotation_logs

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python annotate_clasp_alternatives.py "${COMMON_ARGS[@]}" \
  --clasp-ids $(seq 0 4 99) \
  > annotation_logs/clasp_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
  python annotate_clasp_alternatives.py "${COMMON_ARGS[@]}" \
  --clasp-ids $(seq 1 4 99) \
  > annotation_logs/clasp_gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  python annotate_clasp_alternatives.py "${COMMON_ARGS[@]}" \
  --clasp-ids $(seq 2 4 99) \
  > annotation_logs/clasp_gpu2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
  python annotate_clasp_alternatives.py "${COMMON_ARGS[@]}" \
  --clasp-ids $(seq 3 4 99) \
  > annotation_logs/clasp_gpu3.log 2>&1 &

wait
echo "All CLASP annotation processes finished."



# python build_gum_transition_costs.py \
#   --gum gum \
#   --output gum_transition_costs.json

# python calculate_entity_iv.py \
#   annotated_ns_alternatives annotated_clasp_alternatives \
#   --transition-costs gum_transition_costs.json \
#  --out-dir iv_result
