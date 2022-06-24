import argparse
import os
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

from const import DATA_DIR, SCALER_FILE, CELL_FEAT_FILE

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', type=str,
                    help="List of dirs that contains embeddings.npy of cell lines")
args = parser.parse_args()

paths = dict()
embeddings = []
for d in args.dirs:
    f = os.path.join(DATA_DIR, d, 'embeddings.npy')
    emb = np.load(f)
    embeddings.append(emb)
embedding = np.concatenate(embeddings, axis=1)
scaler = StandardScaler().fit(embedding)
embedding = scaler.transform(embedding)
# joblib.dump(scaler, SCALER_FILE)
# print("Saved", SCALER_FILE)
np.save(CELL_FEAT_FILE, embeddings)
print("Saved", CELL_FEAT_FILE)
