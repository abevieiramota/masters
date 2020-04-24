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


params = {
        'dp_scorer': 'markov',
        'dp_scorer_n': 4,
        'max_dp': 100,

        'sa_scorer': 'markov',
        'sa_scorer_n': 4,
        'max_sa': 100,

        'tems_lm_name': 'markov',
        'tems_lm_n': 5,
        'max_tems': 4,
        'fallback_template': 'jjt',

        'referrer': 'abe',
        'referrer_lm_n': 3,
        'max_refs': 5,

        'txs_lm_name': 'markov',
        'txs_lm_n': 5,
        
        'lp_n': 0,
        'lp_a': 0
}

tgp = make_model(params, ('train', 'dev'))

model_name = 'el_abzao'

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
print(results)
