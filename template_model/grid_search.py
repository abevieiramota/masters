# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset, Entry
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
import os
from plain_experimento import make_model
import sys
import glob
import pickle
from itertools import islice
from sklearn.model_selection import ParameterGrid
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, evaluate_system


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMS_LM_TEXT_FILENAME = 'tems_lm_text.txt'
TEMS_LM_MODEL_FILENAME = 'tems_lm.arpa'
TXS_LM_TEXT_FILENAME = 'txs_lm_text.txt'
TXS_LM_MODEL_FILENAME = 'txs_lm.arpa'
TEMPLATE_DB_FILENAME = 'template_db'
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')


grid =  [
    {
        'tems_lm_name': ['lower'],
        'txs_lm_name': ['lower'],
        'tems_lm_n': [3],
        'tems_lm_bos': [True],
        'tems_lm_eos': [True],
        'tems_lm_preprocess_input': ['lower'],
        'txs_lm_preprocess_input': ['lower'],
        'txs_lm_n': [3],
        'txs_lm_bos': [True],
        'txs_lm_eos': [True],
        'dp_scorer': ['markov_n=3'],
        'sa_scorer': ['ltr_lasso'],
        'max_dp': [2],
        'max_sa': [2],
        'max_tems': [2],
        'max_refs': [1],
        'fallback_template': ['jjt'],
        'referrer': ['abe'],
        'referrer_lm_n': [3],
        'model_name': ['tems3_txs3']
    },
    {
        'tems_lm_name': ['lower'],
        'txs_lm_name': ['lower'],
        'tems_lm_n': [6],
        'tems_lm_bos': [True],
        'tems_lm_eos': [True],
        'tems_lm_preprocess_input': ['lower'],
        'txs_lm_preprocess_input': ['lower'],
        'txs_lm_n': [6],
        'txs_lm_bos': [True],
        'txs_lm_eos': [True],
        'dp_scorer': ['markov_n=3'],
        'sa_scorer': ['ltr_lasso'],
        'max_dp': [2],
        'max_sa': [2],
        'max_tems': [2],
        'max_refs': [1],
        'fallback_template': ['jjt'],
        'referrer': ['abe'],
        'referrer_lm_n': [6],
        'model_name': ['tems6_txs6']
    }
]
# best n=3
grid =  [
    {
        'tems_lm_name': ['lower'],
        'txs_lm_name': ['lower'],
        'tems_lm_n': [3],
        'tems_lm_bos': [True],
        'tems_lm_eos': [True],
        'tems_lm_preprocess_input': ['lower'],
        'txs_lm_preprocess_input': ['lower'],
        'txs_lm_n': [3],
        'txs_lm_bos': [True],
        'txs_lm_eos': [True],
        'dp_scorer': ['markov_n=3', 'random'],
        'sa_scorer': ['ltr_lasso', 'random'],
        'max_dp': [4],
        'max_sa': [4],
        'max_tems': [4],
        'max_refs': [4],
        'fallback_template': ['jjt'],
        'referrer': ['abe'],
        'referrer_lm_n': [3],
        'max_texts': [1, 10, 100, 1000, 10000]
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

    with open(texts_outpath, 'w', encoding='utf-8') as f:
        template_infos = []
        for i, e in enumerate(dev):
            text = tgp.make_text(e)
            f.write(f'{text}\n')
            if i % 100 == 0:
                print(i)

    preprocess_model_to_evaluate(texts_outpath, 'dev')

    results = evaluate_system(model_name,
                              'dev',
                              ['all-cat'],
                              methods=['bleu', 'meteor', 'ter'])

    print(model_name)
    print(results)
