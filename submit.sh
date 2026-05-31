#!/bin/bash
#BSUB -J train_baseline
#BSUB -q short-gpu
#BSUB -gpu "num=1:j_exclusive=yes:gmem=8G:gmodel=NVIDIAA40"
#BSUB -R "rusage[mem=10000]"
#BSUB -R "affinity[thread*10]"
#BSUB -o /home/projects/roilab/anam/phD/tokenization_genomes/model_results/train_%J.out
#BSUB -e /home/projects/roilab/anam/phD/tokenization_genomes/model_results/train_%J.err

# Do NOT load Python/CUDA modules — cu126 torch bundles its own CUDA runtime,
# and EasyBuild Python modules contaminate the conda env's stdlib path.

# Isolate the conda env's Python completely
unset PYTHONPATH PYTHONSTARTUP
export PYTHONNOUSERSITE=1
export PYTHONHOME=/home/projects/roilab/anam/envs/meta

PY=/home/projects/roilab/anam/envs/meta/bin/python

echo "=== Environment check ==="
$PY -c "import sys; print('exec:', sys.executable)"
$PY -c "import sys; print('stdlib:', [p for p in sys.path if 'python3.11' in p])"
$PY -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"
$PY -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo "========================="

BLOCK=${BLOCK:-block16}
LEVEL=${LEVEL:-genus}
LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-1000}
NOTES=${NOTES:-""}
MIN_GEN=${MIN_GEN:-10}

# Auto-named output folder from parameters
LR_TAG=$(echo "$LR" | sed 's/\./_/g; s/e-/em/g')
RUN_NAME="${BLOCK}__${LEVEL}__lr${LR_TAG}__ep${EPOCHS}"
[ -n "$NOTES" ] && RUN_NAME="${RUN_NAME}__${NOTES}"

BASE=/home/projects/roilab/anam/phD/tokenization_genomes
OUT_DIR=${BASE}/model_results/${RUN_NAME}
SCRIPT_DIR=/home/projects/roilab/anam/phD/tokenization_genomes/tokotax

mkdir -p "$OUT_DIR"

# move logs into run folder once job starts
cp /proc/self/fd/1 "$OUT_DIR/train_${LSB_JOBID}.out" 2>/dev/null || true
echo "Training with embedding layer: ${BLOCK}"
echo "Results -> ${OUT_DIR}"

$PY ${SCRIPT_DIR}/train_baseline.py \
    --data_dir          ${BASE}/model_files \
    --out_dir           ${OUT_DIR} \
    --embedding_file    embeddings_${BLOCK}.pt \
    --level             ${LEVEL} \
    --min_genomes       ${MIN_GEN} \
    --epochs            ${EPOCHS} \
    --batch_size        64 \
    --lr                ${LR} \
    --adapter_rank      8 \
    --freq_emb_dim      1024