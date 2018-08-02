# kivisaari_2018
This repository contains code and other material relating to Kivisaari et al. 2018: The brain reconstructs bits of 
information into rich meaningful representations


Authors:
Sasa Kivisaari sasa.kivisaari@aalto.fi
Marijn van Vliet marijn.vanvliet@aalto.fi


The zero-shot learning code requires Python 3 environment and the following
packages:

numpy
scipy
pandas
scikit-learn
progressbar



A brief description of files in this repository:

zero_shot_decoding.py - This contains the actual machine learning code. 

ridge.py - Linear regression using a regularization parameter (ridge regression).
Adapted from the scikit-learn 0.18 code by Marijn van Vliet to suppert the
alpha_per_target parameter.

stability_selection.py - This script is for selecting voxels with highest 
correlation across trials

stimulus_list.pdf - List of stimuli used in the experiment. 

