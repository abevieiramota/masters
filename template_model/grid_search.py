# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        Entry
)
import os
from plain_experimento import (
        TextGenerationPipeline
)
from pretrained_models import (
        load_referrer,
        load_template_selection_lm,
        load_text_selection_lm,
        load_template_db,
        load_discourse_planning,
        load_sentence_aggregation,
        load_template_fallback,
        load_preprocessing
)
from itertools import product
import sys
import glob
import pickle
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
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

grid = {
        'tems_lm_name': ['lower'],
        'tems_lm_n': [3, 6],
        'tems_lm_bos': [False],
        'tems_lm_eos': [False],
        'tems_lm_preprocess_input': ['lower'],
        'txs_lm_preprocess_input': ['lower'],
        'txs_lm_name': ['lower'],
        'txs_lm_n': [3, 6],
        'txs_lm_bos': [False],
        'txs_lm_eos': [False],
        'dp_scorer': ['random', 'ltr_lasso'],
        'sa_scorer': ['random', 'ltr_lasso'],
        'max_dp': [3],
        'max_sa': [4],
        'max_tems': [5],
        'fallback_template': ['jjt'],
        'referrer': ['counter']
}

# already ran
models = [os.path.basename(p) for p in glob.glob(f'../data/models/dev/*')]
already_ran_params = []
for model in models:
    with open(f'../data/models/dev/{model}/params.pkl', 'rb') as f:
        params = pickle.load(f)
        already_ran_params.append(params)


for params in [{k: v for k, v in zip(grid.keys(), x)}
               for x in product(*grid.values())]:

    if params in already_ran_params:
        continue

    model_name = hash(tuple(params.items()))
    # 1. Grid Search
    # 1.1. Language Models
    # 1.1.1 Template Selection Language Model
    tems_lm = load_template_selection_lm(['train'],
                                         params['tems_lm_n'],
                                         params['tems_lm_name'])
    tems_lm_preprocess_input = load_preprocessing(
            params['tems_lm_preprocess_input'])
    # 1.1.2 Text Selection Language Model
    txs_lm = load_text_selection_lm(['train'],
                                    params['txs_lm_n'],
                                    params['txs_lm_name'])
    txs_lm_preprocess_input = load_preprocessing(
            params['txs_lm_preprocess_input'])
    # 1.2 Template database
    template_db = load_template_db(['train'])
    # 1.3 Referring Expression Generation
    referrer = load_referrer(['train'], params['referrer'])
    # 1.4 Discourse Planning
    dp_scorer = load_discourse_planning(['train'],
                                        params['dp_scorer'])
    # 1.5 Sentence Aggregation
    sa_scorer = load_sentence_aggregation(['train'],
                                          params['sa_scorer'])
    # 1.6 Template Fallback
    fallback_template = load_template_fallback(['train'],
                                               params['fallback_template'])

    # Model
    tgp = TextGenerationPipeline(
            template_db,
            tems_lm,
            params['tems_lm_bos'],
            params['tems_lm_eos'],
            tems_lm_preprocess_input,
            txs_lm,
            params['txs_lm_bos'],
            params['txs_lm_eos'],
            txs_lm_preprocess_input,
            dp_scorer,
            sa_scorer,
            params['max_dp'],
            params['max_sa'],
            params['max_tems'],
            fallback_template,
            referrer)

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
