# Running the Baselines
## DeepDDS
### Installation
You should first install RDKIT package.
```
conda install -y -c conda-forge rdkit=2020.09.5
```
Then, install other packages we needed, such as Pytorch Geometric:
```
pip install scipy
TORCH=1.7.1 && CUDA=cu110 && \
pip install torch-scatter --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-sparse --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-cluster --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-spline-conv --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-geometric
```

### Dataset
To create your own dataset, you should prepare the SMILES file like `drug_smiles_new.csv`, train and test dds files like `train_fold0.csv` and `test_fold0.csv`, as well as the cell line feature file like `cell_features_expression_new.csv`.


Then, run the following:
```
python baselines/DeepDDs-master/creat_data_DC.py \
       --cellfile $CELLDIR \
       --savedir $SAVEDIR \
       --root $ROOT
```

### Training
Just run `training_GCN.py` file after modifying the path accordingly.

## GraphSynergy
### Dataset
Create the dataset for Graphsynergy by running `baselines/GraphSynergy-master/preprocessing/data_transform.py` file. 

### Training
Run the script like the following.
```bash
python baselines/GraphSynergy-master/train.py --config baselines/GraphSynergy-master/config/trans/NatureData_config_fold0.json
```

## PRODeepSyn
### Dataset
Running file `baselines/PRODeepSyn-main/preprocessing/dds_dataset_trasform.py`, `baselines/PRODeepSyn-main/preprocessing/dds_dataset_transform_leave_comb.py` and `baselines/PRODeepSyn-main/preprocessing/dds_dataset_transform_leave_cell.py`.

It should be note that you should download two files at https://figshare.com/projects/Pisces/150657, i.e. `node_features.npy` and `ppi.csv`, and then put them into the path `baselines/PRODeepSyn-main/cell/data_ours`.
### Training
Just refer to `baselines/PRODeepSyn-main/README.md`.

## DeepSynergy
### Dataset
Running `baselines/DeepSynergy-master/preprocessing/data_generate_tpm.py`.
### Training
Runing `baselines/DeepSynergy-master/train.py`.

## AuDNNSynergy
### Dataset
The data files are copied from that used in `PRODeepSyn`.
### Training
```bash
bash baselines/AuDNNsynergy/predictor/run.sh
```