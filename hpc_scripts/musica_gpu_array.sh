#!/bin/bash
#SBATCH --job-name=mace_array
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err

# Usage (from the directory containing the run folders):
#   bash musica_gpu_array.sh            -> submits one array job, one task per folder
#   bash musica_gpu_array.sh 'MD_NVT_T*' -> only folders matching the pattern
# Optional: MAXPAR=3 bash musica_gpu_array.sh  -> at most 3 tasks running at once

PATTERN="${1:-*}"

# ---- Submission mode: not inside Slurm yet -> count folders and submit array ----
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    shopt -s nullglob
    DIRS=( $PATTERN/ )
    N=${#DIRS[@]}
    if [ "$N" -eq 0 ]; then
        echo "No folders matching '$PATTERN' found." >&2
        exit 1
    fi
    printf '%s\n' "${DIRS[@]%/}" > dirlist.txt
    echo "Submitting array of $N tasks for:"
    cat dirlist.txt
    ARRAY="0-$((N-1))"
    [ -n "$MAXPAR" ] && ARRAY="${ARRAY}%${MAXPAR}"
    sbatch --array="$ARRAY" "$0" "$PATTERN"
    exit $?
fi

# ---- Task mode: each array task runs in its own folder ----
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBMIT_DIR/dirlist.txt")
if [ -z "$DIR" ] || [ ! -d "$SUBMIT_DIR/$DIR" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no valid folder found (got '$DIR')" >&2
    exit 1
fi
cd "$SUBMIT_DIR/$DIR" || exit 1
echo "Task $SLURM_ARRAY_TASK_ID running in $PWD"

source /data/fs201237/mv26603/software/miniconda3/bin/activate
conda activate /data/fs201237/mv26603/miniconda_envs/dev

echo $SLURM_JOB_NODELIST >> node_info
nvidia-smi -L >> node_info

mkdir -p traj
python mace_nvt.py > "job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
