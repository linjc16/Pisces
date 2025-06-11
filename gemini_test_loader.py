import pandas as pd

def load_test_data():
    combos = pd.read_csv("data/ddses_new.csv")
    drug_smiles_df = pd.read_csv("data/drug_smiles.csv")
    cell_ids_df = pd.read_csv("data/cell_ids.csv")

    # Create mapping
    drug_smiles_map = dict(zip(drug_smiles_df["drug_names"], drug_smiles_df["smiles"]))
    cell_index_map = dict(zip(cell_ids_df["cell_line_names"], cell_ids_df["index"]))

    valid_rows = []
    for _, row in combos.iterrows():
        a, b, cell = row["anchor_names"], row["library_names"], row["cell_line_names"]
        if a in drug_smiles_map and b in drug_smiles_map and cell in cell_index_map:
            valid_rows.append({
                "drug_a_name": a, 
                "drug_b_name": b,
                "smiles_a": drug_smiles_map[a],
                "smiles_b": drug_smiles_map[b],
                "cell_idx": cell_index_map[cell],
                "label": row["labels"]
            })

    return valid_rows

