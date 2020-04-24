# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset, Entry
import os
from plain_experimento import make_model
import sys
import glob
import pickle
from itertools import islice
from sklearn.model_selection import ParameterGrid
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, evaluate_system


grid =  [
    {
        'tems_lm_name': ['markov'],
        'txs_lm_name': ['markov'],
        'tems_lm_n': [3, 4, 5],
        'txs_lm_n': [3, 4, 5],
        'dp_scorer': ['markov'],
        'dp_scorer_n': [3, 4, 5],
        'sa_scorer': ['markov'],
        'sa_scorer_n': [3, 4, 5],
        'max_dp': [3],
        'max_sa': [3],
        'max_tems': [3],
        'max_refs': [3],
        'fallback_template': ['jjt'],
        'referrer': ['abe'],
        'referrer_lm_n': [3, 4, 5],
        'lp_n': [0],
        'lp_a': [0]
    }
]

# already ran
models = [os.path.basename(p) for p in glob.glob(f'../data/models/dev/*')]
already_ran_params = []
for model in models:
    with open(f'../data/models/dev/{model}/params.pkl', 'rb') as f:
        params = pickle.load(f)
        already_ran_params.append(params)

dev = load_dataset('dev')

for params in ParameterGrid(grid):

    if params in already_ran_params:
        continue

    for param, value in params.items():
        print(param, ':', value)

    if 'model_name' not in params:
        model_name = hash(tuple(params.items()))
    else:
        model_name = params['model_name']

    tgp = make_model(params, ('train',))

    # create model folder
    outdir = f"../data/models/dev/{model_name}"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # save its hyperparameters
    parameters_outpath = (f"../data/models/dev/{model_name}/"
                          f"params.pkl")

    with open(parameters_outpath, 'wb') as f:
        pickle.dump(params, f)

    texts_outpath = (f"../data/models/dev/{model_name}/"
                     f"{model_name}.txt")

    import time 

    ini = time.time()
    with open(texts_outpath, 'w', encoding='utf-8') as f:
        template_infos = []
        for i, e in enumerate(dev):
            text = tgp.make_text(e)
            f.write(f'{text}\n')
            if i % 100 == 0:
                print(i)
    end = time.time()
    elapsed_time = end - ini 
    with open((f"../data/models/dev/{model_name}/"
               f"elapsed_time.txt"), 'w') as f:
        f.write(f'{elapsed_time}')

    preprocess_model_to_evaluate(texts_outpath, 'dev')

    results = evaluate_system(model_name,
                              'dev',
                              ['all-cat'],
                              methods=['bleu', 'meteor', 'ter'])

    print(model_name)
    print(results)
