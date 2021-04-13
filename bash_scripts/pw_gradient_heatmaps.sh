#!/bin/bash
#SBATCH --partition=volta-hp
#SBATCH --gpus=1
#SBATCH --mem=64G



set -eux

ch-run imgdir2/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 gradient_heatmap.py
EOF