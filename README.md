# kivisaari2019
This repository contains code and other material relating to Kivisaari et al. 2019: Reconstructing meaning from bits of 
information


## Authors:

Marijn van Vliet marijn.vanvliet@aalto.fi
Sasa Kivisaari sasa.kivisaari@aalto.fi



## A brief description of files in this repository:

*zero_shot/zero_shot_decoding.py* - This contains the actual machine learning code. 

*zero_shot/ridge.py* - Linear regression using a regularization parameter (ridge regression).
Adapted from the scikit-learn 0.18 code by Marijn van Vliet to suppert the
alpha_per_target parameter.

*zero_shot/stability_selection.py* - This script is for selecting voxels with highest 
correlation across trials

*stimuli/stimulus_list.xlsx* - List of stimuli used in the experiment and translations thereof.

*stimuli/clues_block<A/B/C/D/E/F>.txt* - All clue triplets in the experiment.

*behavioral/behavioral_data.xslx* - Behavioral accuracies in the Guessing game task by target item and block. 
