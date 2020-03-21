# -*- coding: utf-8 -*-
from plain_experimento import make_model
import os
import pickle
from reading_thiagos_templates import load_shared_task_test, Entry
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
import sys
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, evaluate_system


params = {
        'tems_lm_name': 'lower',
        'txs_lm_name': 'lower',
        'tems_lm_n': 3,
        'txs_lm_n': 3,
        'dp_scorer': 'markov',
        'dp_scorer_n': 3,
        'sa_scorer': 'markov',
        'sa_scorer_n': 3,
        'max_dp': 1,
        'max_sa': 1,
        'max_tems': 1,
        'max_refs': 1,
        'fallback_template': 'jjt',
        'referrer': 'abe',
        'referrer_lm_n': 3,
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
parameters_outpath = (f"../data/models/test/{model_name}/"
                      f"params.pkl")

with open(parameters_outpath, 'wb') as f:
    pickle.dump(params, f)

texts_outpath = (f"../data/models/test/{model_name}/"
                 f"{model_name}.txt")

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
with open((f"../data/models/test/{model_name}/"
            f"elapsed_time.txt"), 'w') as f:
    f.write(f'{elapsed_time}')

preprocess_model_to_evaluate(texts_outpath, 'test')

results = evaluate_system(model_name, 'test', ['old-cat', 'all-cat'])

print(model_name)
print(results)
