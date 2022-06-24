# PRODeepSyn:Integrating Protein-Protein Interaction Network with Omics Data to Predict Anticancer Synergistic Drug Combinations



## Start

large data files are maintain with git lfs, see  [train.py · Issue #1 · TOJSSE-iData/PRODeepSyn · GitHub](https://github.com/TOJSSE-iData/PRODeepSyn/issues/1) for details

~~~bash
git clone  # this repo
cd ProDeepSyn
git lfs install && git-lfs pull  # git lfs
pip install virtualenv
virtualenv venv --no-site-packages --python=python3.
# ...
source venv/bin/activate
pip install -r requirements.txt
~~~

## Run

~~~bash
# construct drug features
# the constructed drug features already exists
cd drug
python gen_feat.py
cd ..

# construct cell line embeddings
# the constructed cell line embeddings already exists
cd cell
## construct cell line embeddings with gene expression data, use gpu if any
python train.py target_ge.npy nodes_ge.npy --suffix sample # --gpu 0
## ...
## construct cell line embeddings with mutation data, use gpu if any
python train.py target_mut.npy nodes_mut.npy --suffix sample # --gpu 0
## ...
python gen_feat.py mdl_ge_128x384_sample mdl_mut_128x384_sample
## ...
cd ..

# 5-fold nested cross-validation, use gpu if any
cd predictor
python cross_validate.py --batch 512 --hidden 2048 4096 8192 --lr 0.001 0.0001 0.00001 --suffix sample # --gpu 0
## ...
## eval cv
python eval_cv.py cv_sample
## ...
~~~


## Cite

~~~
To be added...
~~~

