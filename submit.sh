#!/bin/bash

# ── Experiment config ────────────────────────────────────────────────────────
BLOCK=${BLOCK:-block16}
LEVEL=${LEVEL:-genus}
MIN_GENOMES=${MIN_GENOMES:-15}
EPOCHS=${EPOCHS:-300}
LR=${LR:-1e-3}
BATCH_SIZE=${BATCH_SIZE:-64}
ADAPTER_RANK=${ADAPTER_RANK:-8}
FREQ_EMB_DIM=${FREQ_EMB_DIM:-1024}
NOTES=${NOTES:-""}          # optional free-text tag, e.g. NOTES=dropout_test

# ── Derived naming ──────────────────────────────────────────────────────────
# Format: <block>__<level>__lr<lr>__ep<epochs>[__<notes>]
LR_TAG=$(echo $LR | sed 's/\./_/g; s/1e-3/1e3/g')   # 0.001 → 0_001, 1e-3 → 1e3
RUN_NAME="${BLOCK}__${LEVEL}__lr${LR_TAG}__ep${EPOCHS}"
[ -n "$NOTES" ] && RUN_NAME="${RUN_NAME}__${NOTES}"

BASE=/home/projects/roilab/anam/phD/tokenization_genomes
OUT_DIR=${BASE}/model_results/${RUN_NAME}
LOG_DIR=${OUT_DIR}/logs

mkdir -p "$LOG_DIR"

# ── Dynamic BSub submission ─────────────────────────────────────────────────
bsub \
  -J "train_${RUN_NAME}" \
  -q short-gpu \
  -gpu "num=1:j_exclusive=yes:gmem=8G:gmodel=NVIDIAA40" \
  -R "rusage[mem=10000]" \
  -R "affinity[thread*10]" \
  -o "${LOG_DIR}/train_%J.out" \
  -e "${LOG_DIR}/train_%J.err" \
  bash -c "
    unset PYTHONPATH PYTHONSTARTUP
    export PYTHONNOUSERSITE=1
    export PYTHONHOME=/home/projects/roilab/anam/envs/meta
    PY=/home/projects/roilab/anam/envs/meta/bin/python

    echo '=== Run: ${RUN_NAME} ==='
    \$PY -c 'import torch; print(\"torch\", torch.__version__, \"| cuda:\", torch.cuda.is_available())'
    \$PY -c 'import torch; print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")'
    echo '================================'

    \$PY ${BASE}/model_scripts/train_baseline.py \
        --data_dir   ${BASE}/model_files \
        --out_dir    ${OUT_DIR} \
        --embedding_file embeddings_${BLOCK}.pt \
        --level      ${LEVEL} \
        --min_genomes ${MIN_GENOMES} \
        --epochs     ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr         ${LR} \
        --adapter_rank ${ADAPTER_RANK} \
        --freq_emb_dim ${FREQ_EMB_DIM}
  "

echo "Submitted: ${RUN_NAME}"
echo "Results → ${OUT_DIR}"
echo "Logs    → ${LOG_DIR}"