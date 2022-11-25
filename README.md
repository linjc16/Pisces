# Pisces: A combo-wise contrastive learning approach to synergistic drug combination prediction
This repository is the official implementation of [Pisces: A combo-wise contrastive learning approach to synergistic drug combination prediction](https://www.biorxiv.org/content/10.1101/2022.11.21.517439v1). Our work has been accepted by RECOMB 2023. The code is originally forked from [Fairseq](https://github.com/pytorch/fairseq) and [DVMP](https://github.com/microsoft/DVMP).

## Requirements and Installation
* PyTorch version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

You can build the [Dockerfile](Dockerfile) or use the docker image `teslazhu/pretrainmol36:latest`.

To install the code from source
```
git clone https://github.com/linjc16/Pisces.git

pip install fairseq
pip uninstall -y fairseq 

pip install ninja
python setup.py build_ext --inplace
```

# Getting Started
## Dataset
Refer to [this file](preprocess/README.md).

## Data Preprocessing
We evaluate our models on the dataset above. `dds/scripts/train_trans/data_process`, `dds/scripts/train_leave_comb/data_process` and `dds/scripts/train_leave_cell/data_process` are folders for preprocessing of `5-fold CV`, `Stratified CV for drug combinations`, and `Stratified CV for cell lines` settings respectively. To generate the binary data for `fairseq`, take the `5-fold CV` setting (fold 0) as an example, run
```
python dds/scripts/train_trans/data_process/split_trans.py

bash dds/scripts/train_trans/data_process/run_process_trans.sh fold0

bash dds/scripts/train_trans/data_process/run_binarize_trans.sh
```

Note that you need to change the file paths accordingly. More original data can be found [here](https://figshare.com/projects/Pisces/150657).

## Training and Test
All training and test scripts can be seen in `dds/scripts`. For instance,
```
bash dds/scripts/train_trans/run_dv_ppiv2_cons_tri.sh fold0 5e-5 0.01

bash dds/scripts/train_trans/inference/inf_dv_ppiv2_cons_tri.sh fold0 5e-5 0.01
```

## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
