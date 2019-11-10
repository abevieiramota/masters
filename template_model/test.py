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
        'tems_lm_bos': False,
        'tems_lm_eos': False,
        'tems_lm_preprocess_input': 'lower',
        'txs_lm_preprocess_input': 'lower',
        'txs_lm_n': 3,
        'txs_lm_bos': False,
        'txs_lm_eos': False,
        'dp_scorer': 'ltr_lasso',
        'sa_scorer': 'ltr_lasso',
        'max_dp': 2,
        'max_sa': 2,
        'max_tems': 1,
        'max_refs': 1,
        'fallback_template': 'jjt',
        'referrer': 'abe'
}

tgp = make_model(params, ('train', 'dev'))

model_name = 'abe2'

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
#templates_info_outpath = (f'../data/models/test/{model_name}/'
#                          f'templates_info.pkl')

test = load_shared_task_test()

with open(texts_outpath, 'w', encoding='utf-8') as f:
    template_infos = []
    for i, e in enumerate(test):
        text = tgp.make_text(e)
        #template_infos.append(template_info)
        f.write(f'{text}\n')
        if i % 100 == 0:
            print(i)

#with open(templates_info_outpath, 'wb') as ft:
#    pickle.dump(template_infos, ft)

preprocess_model_to_evaluate(texts_outpath, 'test')

results = evaluate_system(model_name, 'test')

print(model_name)
print(results)
