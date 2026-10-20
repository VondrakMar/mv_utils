#!/bin/bash
# Create new run folders from a template folder, filling in the "..." placeholders
# of mace_sspd.py (T_BASE, N_STEPS_SSPD, N_STEPS_LANGEVIN, TIME_STEP, MODEL_PATH) and the
# SEED / T_HIGH of every run*/submit_mus.sh.
#
# Usage: ./make_sspd_runs.sh SRC_DIR SEEDS THIGHS TBASES NSTEPS_SSPD NSTEPS_LANGEVIN TIME_STEP MODEL [--submit]
#   SEEDS THIGHS TBASES NSTEPS_SSPD NSTEPS_LANGEVIN TIME_STEP MODEL may each be a comma-separated list;
#   one folder is created per combination.
# Example: ./make_sspd_runs.sh template 12,13 3000,4000 700 5000 5000 0.00005 MACE_Oct12_com1.model
#   -> seed12_Thigh3000_Tbase700_sspd5000_lang5000_dt0.00005_MACE_Oct12_com1, ... (4 folders)
# MODEL is a file that lives next to the template folder (in its parent directory) and
# is not copied; MODEL_PATH becomes ../../MODEL, relative to the run*/ folders.
# The folder name uses the model basename without extension.
# With --submit, sbatch submit_mus.sh is run in every run* folder.

set -euo pipefail

if [ $# -lt 8 ]; then
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

SRC="${1%/}"
IFS=',' read -r -a SEEDS  <<< "$2"
IFS=',' read -r -a THIGHS <<< "$3"
IFS=',' read -r -a TBASES <<< "$4"
IFS=',' read -r -a NSSPDS <<< "$5"
IFS=',' read -r -a NLANGS <<< "$6"
IFS=',' read -r -a DTS    <<< "$7"
IFS=',' read -r -a MODELS <<< "$8"
SUBMIT=0
[ "${9:-}" = "--submit" ] && SUBMIT=1

[ -d "$SRC" ] || { echo "Source folder not found: $SRC" >&2; exit 1; }
[ -f "$SRC/mace_sspd.py" ] || { echo "No mace_sspd.py in $SRC" >&2; exit 1; }
PARENT="$(dirname "$SRC")"

for seed in "${SEEDS[@]}"; do
for th in "${THIGHS[@]}"; do
for tb in "${TBASES[@]}"; do
for nssp in "${NSSPDS[@]}"; do
for nlan in "${NLANGS[@]}"; do
for dt in "${DTS[@]}"; do
for model in "${MODELS[@]}"; do
    mname="$(basename "$model")"; mname="${mname%.*}"
    DST="$PARENT/seed${seed}_Thigh${th}_Tbase${tb}_sspd${nssp}_lang${nlan}_dt${dt}_${mname}"
    if [ -e "$DST" ]; then
        echo "SKIP $DST (already exists)"
        continue
    fi
    mkdir "$DST"
    # copy everything, leaving out run leftovers and caches
    rsync -a \
        --exclude '__pycache__' --exclude '.ipynb_checkpoints' \
        --exclude 'job_*.out' --exclude 'job_*.err' \
        --exclude 'node_info' --exclude '*.model' --exclude '*.log' --exclude '*.h5md' --exclude '*.h5' \
        "$SRC"/ "$DST"/

    # fill the placeholders (any current value after '=' is replaced, trailing comment dropped)
    sed -i -E \
        -e "s|^(T_BASE *= *).*|\1${tb}|" \
        -e "s|^(N_STEPS_SSPD *= *).*|\1${nssp}|" \
        -e "s|^(N_STEPS_LANGEVIN *= *).*|\1${nlan}|" \
        -e "s|^(TIME_STEP *= *).*|\1${dt}|" \
        -e "s|^(MODEL_PATH *= *).*|\1\"../../${model}\"|" "$DST/mace_sspd.py"

    n=0
    for f in "$DST"/run*/submit_mus.sh; do
        [ -f "$f" ] || continue
        sed -i -E \
            -e "s/^(export SEED=).*/\1\"${seed}\"/" \
            -e "s/^(export T_HIGH=).*/\1\"${th}\"/" "$f"
        n=$((n + 1))
        if [ "$SUBMIT" -eq 1 ]; then
            (cd "$(dirname "$f")" && sbatch submit_mus.sh)
        fi
    done
    echo "Created $DST ($n runs)"
done; done; done; done; done; done; done
