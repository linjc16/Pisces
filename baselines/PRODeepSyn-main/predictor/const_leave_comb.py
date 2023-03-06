import os

PROJ_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
SUB_PROJ_DIR = os.path.join(PROJ_DIR, 'predictor')
DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data_ours_leave_comb')
DRUG_DATA_DIR = os.path.join(PROJ_DIR, 'drug', 'data_ours')
CELL_DATA_DIR = os.path.join(PROJ_DIR, 'cell', 'data_ours')
OUTPUT_DIR = 'baselines/PRODeepSyn-main/predictor/data_ours_leave_comb/output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

fold_id = 4
SYNERGY_FILE = os.path.join(DATA_DIR, f'synergy_fold{fold_id}.tsv')

DRUG_FEAT_FILE = os.path.join(DRUG_DATA_DIR, 'drug_feat.npy')
DRUG2ID_FILE = os.path.join(DRUG_DATA_DIR, 'drug2id.tsv')
CELL_FEAT_FILE = os.path.join(CELL_DATA_DIR, 'cell_feat.npy')
CELL2ID_FILE = os.path.join(CELL_DATA_DIR, 'cell2id.tsv')




