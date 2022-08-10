#!/bin/bash
# puddin_setup.sh

eval "$(conda shell.bash hook)"

conda env create -f puddin_env.yml
conda activate puddin

echo "pytorch ==1.4.0" > ~/anaconda3/envs/puddin/conda-meta/pinned

conda update --all
