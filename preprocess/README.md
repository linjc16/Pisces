# Data Preprocessing
## Generating from Original Dataset
Our dataset are based on datasets from the paper [Effective drug combinations in breast, colon and pancreatic cancer cells](https://www.nature.com/articles/s41586-022-04437-2.epdf?sharing_token=KiodKI9Z3HAT6hV7ZpH_ZtRgN0jAjWel9jnR3ZoTv0PUitZPf3xDEjFaijADzqtEPBs0FK0ZLA545ugwbt_5IJf_tStBEKsGx8gnH_FvlNf-8Gj3GRCCTFbtk-iyb5R_BF2YOr_e_Iom7FC1eocZQM9DNDSVJqATv1AQZe4aB6M%3D). The dataset can be downloaded [here](https://gdsc-combinations.depmap.sanger.ac.uk/).

From the original dataset, we can generate the drug-drug synergy file `ddses.csv` and drug SMILES file `drug_smiles.csv`. Run
```
python preprocess/data_preprocessing.py
```
Note that you should change the `savedir` and the original dataset path accourdingly to your local path.

## Drug Features
For drug features, we adopt RDKit (version 2020.09.5) to obtain the fingerprints and descriptors from the SMILES. Run the script
```
python preprocess/drug_fingerprints.py
python preprocess/drug_descriptors.py
```
## Cell Line Features
We calculate the cell line features based on gene expression features.
### Gene Expression
Gene expression data can be downloaded [here](https://cellmodelpassports.sanger.ac.uk/downloads). We use the RNA-Seq file `all RNA-Seq processed data`. TheN generate cell line features by running
```
python preprocess/cell_read_count.py
python preprocess/cell_tpm.py
```

## Protein-protein Networks