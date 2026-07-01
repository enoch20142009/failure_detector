#!/bin/bash
#PBS -P ka69
#PBS -q copyq
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=60GB
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/ka69+scratch/ka69
#PBS -l wd
#PBS -j oe
#
# Extract RLBench-Fail TRAIN (paulpacaud/rlbenchfail_train_dataset, 12,358 samples)
# into loose start/end image pairs for the Guardian loader.
#
# Guardian recipe: 4 sim cams (start+end), TRAIN subsamples 1 view ~50% of the
# time ("50/50 loose views") -> --full_view_probability 0.5, --max_views 4.
# Avg ~6 inodes/sample => ~74k inodes in the scratch output.
#
# FAST + INODE-SAFE: streaming individual members out of the 8 GiB gzip tar is
# O(n^2) (gzip is not seekable) and previously timed out at walltime. Instead we
# unpack the whole tar ONCE to node-local $PBS_JOBFS (fast sequential read, and
# its ~132k intermediate inodes do NOT count against the scratch quota), then
# read loose files from there and copy only the ~74k SELECTED images to scratch.
#
# copyq has internet (compute nodes do not) + per-node SSD jobfs.
#
#   qsub gadi_scripts/ov2_routing/23_prepare_rlbench_train_guardian.sh
#
set -euo pipefail

module purge
module load pytorch/2.12.0

cd /scratch/ka69/yc0686/robot_failure_classifier/I-FailSense-main
source .venv-ov2/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/ka69/yc0686/hf_cache
mkdir -p "$HF_HOME"

# Online (copyq has internet) so the metadata JSONL (and tar, if not cached) download.
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

OUT_DIR=/scratch/ka69/yc0686/robot_failure_classifier/guardian_selected/rlbenchfail_train_5050
JOBFS_RECORDS="${PBS_JOBFS}/rlbench_records"
mkdir -p "${JOBFS_RECORDS}"

echo "==== Resolve cached records.tar.gz ===="
TAR=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('paulpacaud/rlbenchfail_train_dataset', repo_type='dataset', filename='records.tar.gz'))")
echo "tar: ${TAR}"

echo "==== Unpack tar -> node-local JOBFS (fast, no scratch inodes): ${JOBFS_RECORDS} ===="
time tar -xzf "${TAR}" -C "${JOBFS_RECORDS}"
echo "Unpacked. Top-level:"; ls "${JOBFS_RECORDS}" | head

echo "==== Extract RLBench-Fail train (Guardian 50/50) from JOBFS -> ${OUT_DIR} ===="
python src/prepare_FailCoT_from_jsonl.py \
  --repo_id paulpacaud/rlbenchfail_train_dataset \
  --config_name metadata_execution \
  --split train \
  --records_dir "${JOBFS_RECORDS}" \
  --output_dir "${OUT_DIR}" \
  --full_view_probability 0.5 \
  --max_views 4 \
  --seed 42 \
  --force

echo "Done. Manifest: ${OUT_DIR}/manifest.json"