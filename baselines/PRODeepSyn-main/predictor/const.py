import os

PROJ_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
SUB_PROJ_DIR = os.path.join(PROJ_DIR, 'predictor')
DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data')
DRUG_DATA_DIR = os.path.join(PROJ_DIR, 'drug', 'data')
CELL_DATA_DIR = os.path.join(PROJ_DIR, 'cell', 'data')
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SYNERGY_FILE = os.path.join(DATA_DIR, 'synergy.tsv')

DRUG_FEAT_FILE = os.path.join(DRUG_DATA_DIR, 'drug_feat.npy')
DRUG2ID_FILE = os.path.join(DRUG_DATA_DIR, 'drug2id.tsv')
CELL_FEAT_FILE = os.path.join(CELL_DATA_DIR, 'cell_feat.npy')
CELL2ID_FILE = os.path.join(CELL_DATA_DIR, 'cell2id.tsv')




