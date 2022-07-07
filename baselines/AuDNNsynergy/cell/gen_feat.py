import numpy as np
import pdb

embs = np.load('baselines/AuDNNsynergy/cell/data_ours/mdl_exp_2207061646/embeddings.npy')

np.save('baselines/AuDNNsynergy/cell/data_ours/cell_feat.npy', embs)
