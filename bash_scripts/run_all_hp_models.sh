#!/bin/bash

python training.py --kernel_size 1 1 1 1 --dilations 1 1 1 1 --variable 1 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'
python training.py --kernel_size 1 1 1 1 --dilations 1 1 1 1 --variable 0 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'
python training.py --kernel_size 2 2 2 2 --dilations 1 1 1 1 --variable 1 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'
python training.py --kernel_size 2 2 2 2 --dilations 1 1 1 1 --variable 0 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'
python training.py --kernel_size 3 3 3 3 --dilations 1 1 1 1 --variable 1 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'
python training.py --kernel_size 3 3 3 3 --dilations 1 1 1 1 --variable 0 --starting_patient_index 1 --high_pass_train True --model_string='hp_m_'