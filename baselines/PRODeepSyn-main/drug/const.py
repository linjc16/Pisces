import os

SUB_PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data')

DRUG2ID_FILE = os.path.join(DATA_DIR, 'drug2id.tsv')
FP_FILE = os.path.join(DATA_DIR, 'fp256.tsv')
DESC_FILE = os.path.join(DATA_DIR, 'descriptors.tsv')

SCALER_FILE = os.path.join(DATA_DIR, 'scaler.pkl')
DRUG_FEAT_FILE = os.path.join(DATA_DIR, 'drug_feat.npy')
