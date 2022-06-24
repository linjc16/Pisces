import os

SUB_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data')

COO_FILE = os.path.join(DATA_DIR, 'ppi.coo.npy')
NODE_FEAT_FILE = os.path.join(DATA_DIR, 'node_features.npy')

SCALER_FILE = os.path.join(DATA_DIR, 'scaler.pkl')
CELL_FEAT_FILE = os.path.join(DATA_DIR, 'cell_feat.npy')

