# GraphSynergy
This is our PyTorch implementation for the paper:
> GraphSynergy: Network Inspired Deep Learning Model for Anti-Cancer Drug Combination Prediction

# Introduction
GraphSynergy is a new deep learning framework to make explainable synergistic drug combination predictions. GraphSynergy is inspired by the recent network science studies on drug combination identifying task and utilizes Graph Convolutional Network (GCN) and attention module to capture the topological relations of the protein modules of drugs and cancer cell lines in the PPI network.

# Environment Requirement
The code has been tested running under Python 3.7. The required package are as follows:
* pytorch == 1.6.0
* numpy == 1.19.1
* sklearn == 0.23.2
* networkx == 2.5
* pandas == 1.1.2

# Installation
To install the required packages for running GraphSynergy, please use the following command first
```bash
pip install -r requirements.txt
```
If you meet any problems when installing pytorch, please refer to [pytorch official website](https://pytorch.org/)

# Example to Run the Codes
* DrugCombDB
```bash
python train.py --config ./config/DrugCombDB_config.json
```
* Oncology-Screen
```bash
python train.py --config ./config/OncologyScreen_config.json
```

# Dataset
Datasets used in the paper:
* [Protein-Protein Interaction Network](https://www.nature.com/articles/s41467-019-09186-x#Sec23) is a comprehensive human interactome network.
* [Drug-protein Associations](https://www.nature.com/articles/s41467-019-09186-x#Sec23) are based on FDA-approved or clinically investigational drugs.
* [Cell-protein Associations](https://maayanlab.cloud/Harmonizome/dataset/CCLE+Cell+Line+Gene+Expression+Profiles) is harvested from the Cancer Cell Line Encyclopedia.
* [DrugCombDB](http://drugcombdb.denglab.org/main) is a database with the largest number of drug combinations to date.
* [Oncology-Screen](http://www.bioinf.jku.at/software/DeepSynergy/) is an unbiased oncology compound screen datasets.

# Acknowledgement
Acknowledgement and thanks to others for open source work used in this project. Code used in this project is available from the following sources.
1. https://github.com/victoresque/pytorch-template\
   Author: SunQpark   
   Licensed under [MIT License](https://opensource.org/licenses/MIT).

  
