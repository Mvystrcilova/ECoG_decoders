#!/bin/bash
#SBATCH --partition=volta-lp
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=1-00


set -eux

ch-run imgdir2/diplomka_image sh <<EOF
set -eux
cd /home/vystrcilova/ECoG_decoders
python3 training.py --kernel_size 3 3 3 3 --dilations 2 4 8 16 --variable 0 --starting_patient_index 1
EOF