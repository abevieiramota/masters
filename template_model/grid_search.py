# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset, Entry
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
import os
from plain_experimento import make_model
import sys
import glob
import pickle
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
MODEL_NAME_BASE = 'general{}'

grid = [
        {
            #'tems_lm_name': ['inv_lower', 'random', 'lower'],
            #'txs_lm_name': ['inv_lower', 'random', 'lower'],
            'tems_lm_name': ['lower', 'random'],
            'txs_lm_name': ['lower', 'random'],
            'tems_lm_n': [3, 6],
            'tems_lm_bos': [False],
            'tems_lm_eos': [False],
            'tems_lm_preprocess_input': ['lower'],
            'txs_lm_preprocess_input': ['lower'],
            'txs_lm_n': [3, 6],
            'txs_lm_bos': [False],
            'txs_lm_eos': [False],
            #'dp_scorer': ['random', 'inv_ltr_lasso', 'ltr_lasso'],
            #'sa_scorer': ['random', 'inv_ltr_lasso', 'ltr_lasso'],
            'dp_scorer': ['ltr_lasso', 'random'],
            'sa_scorer': ['ltr_lasso', 'random'],
            'max_dp': [2],
            'max_sa': [4],
            'max_tems': [2],
            'fallback_template': ['jjt'],
            'referrer': ['counter', 'preprocess_so']
            #'referrer': ['counter', 'preprocess_so', 'inv_counter']
        }
]

# already ran
models = [os.path.basename(p) for p in glob.glob(f'../data/models/dev/*')]
already_ran_params = []
for model in models:
    with open(f'../data/models/dev/{model}/params.pkl', 'rb') as f:
        params = pickle.load(f)
        already_ran_params.append(params)


for params in ParameterGrid(grid):

    if params in already_ran_params:
        continue

    if 'model_name' not in params:
        model_name = hash(tuple(params.items()))
    else:
        model_name = params['model_name']

    tgp = make_model(params, ['train'])

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

    dev = load_dataset('dev')

    with open(texts_outpath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(dev):
            text = tgp.make_text(e)
            f.write(f'{text}\n')
            if i % 100 == 0:
                print(i)

    preprocess_model_to_evaluate(texts_outpath, 'dev')

    results = evaluate_system(model_name,
                              'dev',
                              methods=['bleu', 'meteor', 'ter'])

    print(model_name)
    print(results)
