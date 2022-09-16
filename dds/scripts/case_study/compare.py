from unittest import result
import pandas as pd
import numpy as np
import pdb

result_deepdds = pd.read_csv('comb_level_results_deepdds.csv', index_col=0)
result_gs = pd.read_csv('comb_level_results_graphSynergy.csv', index_col=0)
result_ours = pd.read_csv('comb_level_results.csv', index_col=0)



diff = []

for i in range(len(result_deepdds)):
    F1_deepdds = result_deepdds.iloc[i, :]['AUPRC']
    F1_gs = result_gs.iloc[i, :]['AUPRC']
    F1_ours = result_ours.iloc[i, :]['AUPRC']

    if result_deepdds.iloc[i, :]['num_pos'] == 0:
        diff.append(0)
        continue

    diff.append(F1_ours - max(F1_gs, F1_deepdds))

diff_df = pd.DataFrame.from_dict(
    {
        'id': result_deepdds['id'].tolist(),
        'anchor_names': result_deepdds['anchor_names'].tolist(),
        'library_names': result_deepdds['library_names'].tolist(),
        'AUPRC_diff': diff
    }
)

diff_df.to_csv('diff.csv', index=False)
    