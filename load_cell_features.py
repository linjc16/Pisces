import pandas as pd
import torch

def load_cell_tpm_features():
    df = pd.read_csv("data/cell_tpm.csv")
    df = df.set_index("cell_line_names")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df

def get_cell_feature_vector(cell_name: str, tpm_df):
    if cell_name not in tpm_df.index:
        raise ValueError(f"Cell line '{cell_name}' not found in TPM dataset.")
    vector = tpm_df.loc[cell_name].values.astype('float32')
    return torch.tensor(vector).unsqueeze(0)
