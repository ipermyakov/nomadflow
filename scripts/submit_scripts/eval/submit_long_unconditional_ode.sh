#!/bin/bash
#$ -S /bin/bash
#$ -o tmp.log
#$ -j y
#$ -r y
#$ -cwd
#$ -q gpu.q
#$ -pe smp 1
#$ -l mem_free=32G
#$ -l h_rt=14:00:00
#$ -l compute_cap=61,gpu_mem=40G

export CUDA_VISIBLE_DEVICES=$SGE_GPU
export GEOMSTATS_BACKEND=pytorch
echo $SGE_GPU
echo $CUDA_VISIBLE_DEVICES

module load Sali cuda

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "${SCRIPT_DIR}/../../../env_vars.sh"

ROOT_DIR=$PWD
cd $1
RUN_DIR=$PWD
cd $ROOT_DIR

# EPOCH=$2
# OUTPREFIX=$3
OUTPREFIX=$2

if [ -n "$3" ]
then
    CHECKPOINTIDX=$3
else
    CHECKPOINTIDX=-1
fi

## generate samples
conda activate ${ENV_NAME}

python sample.py \
    model_dir=$RUN_DIR \
    out_dir=$OUTPREFIX \
    checkpoint_idx=$CHECKPOINTIDX \
    sampler.tasks_yaml=${REPO_ROOT}/sampling/unconditional_sampling/benchmark_300.yaml \
    sampler.batch_size=10 \
    corrupter.sampling_noise_mode=null \
    corrupter.trans_step_size=1 \
    corrupter.rot_step_size=10 \
    corrupter.num_timesteps=200 \

## sample with ProteinMPNN
cd ~/software/ProteinMPNN/scripts
conda activate proteinmpnn

folder_with_pdbs=$OUTPREFIX/samples
output_dir=$OUTPREFIX
path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_fixed_pos=$output_dir"/pmpnn_fixed_pos_dict.jsonl"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 8 \
        --sampling_temp "0.1" \
        --batch_size 8

## submit ESMFold
cd $OUTPREFIX
mkdir esmfold
cd esmfold
bash ${REPO_ROOT}/scripts/analysis/esmfold.sh > esmfold.log

conda activate ${ENV_NAME}
python ${REPO_ROOT}/scripts/analysis/esm_analysis.py --esmlog esmfold.log --folded_folder $PWD --samples ../samples
cd ..
bash ${REPO_ROOT}/scripts/analysis/analysis_suite.sh