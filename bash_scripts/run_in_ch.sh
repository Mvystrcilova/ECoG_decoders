#!/bin/bash
#SBATCH --partition=volta-hp
#SBATCH --gpus=1
#SBATCH --mem=250G


set -eux

ch-run imgdir2/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 layer_passes.py
EOF