#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH --no-requeue


set -eux

ch-run imgdir/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 training.py
EOF