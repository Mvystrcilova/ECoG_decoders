#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=1-00


set -eux

ch-run imgdir2/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 gradient_multigraphs.py --variable=0 --channels='NCH'
EOF