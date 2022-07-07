import pandas as pd
import pdb

dds_df = pd.read_csv('data/ddses.csv', index_col=0)

cell_list = dds_df['cell_line_names'].tolist()
drug1_db_list = dds_df['anchor_names'].tolist()
drug2_db_list = dds_df['library_names'].tolist()
synergy_list = dds_df['labels'].tolist()

dds_df_new = pd.DataFrame.from_dict(
    {
        'cell': cell_list,
        'drug1_db': drug1_db_list,
        'drug2_db': drug2_db_list,
        'synergy': synergy_list})

dds_df_new.to_csv('baselines/GraphSynergy-master/data_ours/drug_combinations.csv', index=False)