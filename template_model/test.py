# -*- coding: utf-8 -*-
from plain_experimento import make_model
import os
import pickle
from reading_thiagos_templates import load_shared_task_test, Entry
import logging
import sys
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, evaluate_system


log_level = sys.argv[1] if len(sys.argv) > 1 else 'INFO'

logging.basicConfig(level=log_level)

n = 3

params = {
        'dp_scorer': 'markov',
        'dp_lm_n': 4,
        'max_dp': n,

        'sa_scorer': 'markov',
        'sa_lm_n': 5,
        'max_sa': n,

        'tems_lm_name': 'markov',
        'tems_lm_n': 3,
        'max_tems': n,
        'fallback_template': 'jjt',
        'lp_tems_n': 0,
        'lp_tems_a': 0,

        'referrer': 'abe',
        'referrer_lm_n': 3,
        'max_refs': n,

        'txs_lm_name': 'markov',
        'txs_lm_n': 5,
        'lp_txs_n': 0,
        'lp_txs_a': 0
}

tgp = make_model(params, ('train', 'dev'))

model_name = '{}{}{}{}{}__{}_{}_{}_{}_lp_tems_{}_{}_lp_txs_{}_{}'.format(
    params['dp_lm_n'],
    params['sa_lm_n'],
    params['tems_lm_n'],
    params['referrer_lm_n'],
    params['txs_lm_n'],
    params['max_dp'], 
    params['max_sa'],   
    params['max_tems'], 
    params['max_refs'], 
    params['lp_tems_n'], 
    params['lp_tems_a'],
    params['lp_txs_n'],
    params['lp_txs_a']
)

# create model folder
outdir = f"../data/models/test/{model_name}"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# save its hyperparameters
parameters_outpath = (f"../data/models/test/{model_name}/params.pkl")

with open(parameters_outpath, 'wb') as f:
    pickle.dump(params, f)

texts_outpath = (f"../data/models/test/{model_name}/{model_name}.txt")

test = load_shared_task_test()

import time 

ini = time.time()

with open(texts_outpath, 'w', encoding='utf-8') as f:
    template_infos = []
    for i, e in enumerate(test):
        text = tgp.make_text(e)
        f.write(f'{text}\n')
        if i % 100 == 0:
            print(i)

end = time.time()
elapsed_time = end - ini 
with open((f"../data/models/test/{model_name}/elapsed_time.txt"), 'w') as f:
    f.write(f'{elapsed_time}')

preprocess_model_to_evaluate(texts_outpath, 'test')

results = evaluate_system(model_name, 'test', ['old-cat', 'all-cat'])

print(model_name)
print(elapsed_time)
print(results)
