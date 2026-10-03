#!/bin/bash
#$ -S /bin/bash
#$ -o OUTFILE.log
#$ -j y
#$ -r y
#$ -R y
#$ -cwd
#$ -q gpu.q
#$ -pe smp NUMGPU
#$ -l mem_free=32G
#$ -l h_rt=TRAINHOURS:00:00
#$ -l compute_cap=61,gpu_mem=40G
#$ -l h=!(qb3-idgpu18)

# if you need to exclude nodes
# -l h=!(qb3-atgpu32)&!(qb3-atgpu33)
source REPOROOT/env_vars.sh

echo $JOB_ID
conda activate ${ENV_NAME}

export CUDA_VISIBLE_DEVICES=$SGE_GPU
export GEOMSTATS_BACKEND=pytorch
echo $SGE_GPU
echo $CUDA_VISIBLE_DEVICES

module load Sali cuda

gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

# magic memory flag
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 2048
cd ${REPO_ROOT}
python train.py "$@" +job_id=${JOB_ID} +num_days=TRAINDAYS
# args = ARGS

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
