#!/bin/bash
#SBATCH --partition=zen4_0768_h100x4  
#SBATCH --qos zen4_0768_h100x4 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --time=1:00:00
#SBATCH --mem=20GB
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Run configuration for mace_sspd.py — overridden per-run by setup_runs_musica.sh
export MACE_HEAD="default"
export SEED="15"
export T_HIGH="4000"

source /data/fs201237/mv26603/software/miniconda3/bin/activate
conda activate /data/fs201237/mv26603/miniconda_envs/rh

echo $SLURM_JOB_NODELIST >> node_info
nvidia-smi -L >> node_info 
cd traj
cp /scratch/fs201237/mv26603/sspd_Oct14/seed12_Thigh4000_sspd2000_lang1000_dt0.000025_MACE_Oct12_com1/ts_convert.py .
for i in *h5md;
do
	python ts_convert.py --input_file $i --output_file ${i%.h5md}.xyz 
done
cd ..
cd steering
python steering.py
