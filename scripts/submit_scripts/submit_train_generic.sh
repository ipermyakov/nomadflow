#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source $SCRIPT_DIR/../../env_vars.sh

mkdir -p ${SCRIPT_DIR}/autogen_scripts

SUBMIT_TIME=$(date +"%y-%m-%d_%H%M%S")
NUMGPU=$1
TRAIN_DAYS=$2
TRAIN_HOURS=$(($2*24))

sed -e "s|OUTFILE|${REPO_ROOT}/outputs/sge_outs/train_${SUBMIT_TIME}|g" \
    -e "s|REPOROOT|${REPO_ROOT}|g" \
    -e "s|NUMGPU|${NUMGPU}|g" \
    -e "s|TRAINHOURS|${TRAIN_HOURS}|g" \
    -e "s|TRAINDAYS|${TRAIN_DAYS}|g" \
    < "${REPO_ROOT}/scripts/submit_scripts/_train_template_generic.sh" \
    > "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"
echo "#ARGS = ${@:3}" >> "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh"

qsub -terse "${REPO_ROOT}/scripts/submit_scripts/autogen_scripts/train_${SUBMIT_TIME}.sh" "${@:3}"
