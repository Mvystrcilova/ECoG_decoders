#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-00


set -eux

ch-run imgdir2/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 gradient_inspections.py --variable='absVel' --prefixes 'm' 'sm' --files=3
EOF