#!/bin/bash
#PBS -P ka69
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=10GB
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -l wd
#PBS -j oe
#
# 1) Unpack records.tar.gz once (already in HF cache on gadi)
# 2) Build Guardian manifest via filesystem copies (fast path)
#
#   qsub gadi_scripts/ov2_routing/23b_unpack_and_prepare_rlbench_train.sh
#
set -euo pipefail

module purge
module load pytorch/2.12.0

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main
source .venv-ov2/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ka69/yc0686/hf_cache
mkdir -p "$HF_HOME"

TAR_BLOB=/scratch/ka69/yc0686/hf_cache/hub/datasets--paulpacaud--rlbenchfail_train_dataset/blobs/957885a46f73b6654d151050b6a5c57f629b61ad1c54dd257b49e9dc8ae7ac3f
UNPACK_ROOT=/scratch/ka69/yc0686/robot_failure_classifier/guardian_selected/rlbenchfail_train_records_unpacked
RECORDS_DIR="${UNPACK_ROOT}/records"
OUT_DIR=/scratch/ka69/yc0686/robot_failure_classifier/guardian_selected/rlbenchfail_train_5050

if [[ ! -f "${TAR_BLOB}" ]]; then
  echo "Downloading records.tar.gz via huggingface_hub..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("paulpacaud/rlbenchfail_train_dataset", "records.tar.gz", repo_type="dataset"))
PY
  TAR_BLOB=$(python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("paulpacaud/rlbenchfail_train_dataset", "records.tar.gz", repo_type="dataset"))
PY
)
fi

DONE_MARKER="${UNPACK_ROOT}/.unpack_complete"
if [[ -f "${DONE_MARKER}" && -d "${RECORDS_DIR}" ]]; then
  echo "==== Records already unpacked: ${RECORDS_DIR} ===="
elif [[ -d "${RECORDS_DIR}" && ! -f "${DONE_MARKER}" ]]; then
  echo "==== Waiting for in-progress unpack to finish (${RECORDS_DIR}) ===="
  while [[ ! -f "${DONE_MARKER}" ]]; do
    sleep 30
  done
else
  echo "==== Unpacking ${TAR_BLOB} -> ${UNPACK_ROOT} ===="
  mkdir -p "${UNPACK_ROOT}"
  tar -xzf "${TAR_BLOB}" -C "${UNPACK_ROOT}"
  touch "${DONE_MARKER}"
fi

echo "==== Fast manifest build -> ${OUT_DIR} ===="
python src/prepare_FailCoT_fast.py \
  --repo_id paulpacaud/rlbenchfail_train_dataset \
  --config_name metadata_execution \
  --split train \
  --records_dir "${RECORDS_DIR}" \
  --output_dir "${OUT_DIR}" \
  --full_view_probability 0.5 \
  --max_views 4 \
  --seed 42

echo "Done. Manifest: ${OUT_DIR}/manifest.json"
