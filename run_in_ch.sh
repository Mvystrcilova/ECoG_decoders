#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=128G


set -eux

ch-run imgdir/diplomka_image sh <<EOF
set -eux
nvidia-smi
cd /home/vystrcilova/ECoG_decoders
python3 training.py
EOF