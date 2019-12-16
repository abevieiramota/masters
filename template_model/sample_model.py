# -*- coding: utf-8 -*-
from plain_experimento import make_model
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from reading_thiagos_templates import Entry, load_shared_task_test, load_test

params = {
        'tems_lm_name': 'lower',
        'txs_lm_name': 'lower',
        'tems_lm_n': 3,
        'tems_lm_bos': True,
        'tems_lm_eos': False,
        'tems_lm_preprocess_input': 'lower',
        'txs_lm_preprocess_input': 'lower',
        'txs_lm_n': 3,
        'txs_lm_bos': True,
        'txs_lm_eos': False,
        'dp_scorer': 'markov_n=3',
        'sa_scorer': 'ltr_lasso',
        'max_dp': 3,
        'max_sa': 3,
        'max_tems': 2,
        'max_refs': 1,
        'fallback_template': 'jjt',
        'referrer': 'abe'
}

tgp = make_model(params, ('train', 'dev'))

#test = load_shared_task_test()
test = load_test()

from random import Random

r = Random(100)

def get():

    i = r.randint(0, 1800)
    print(f'i = {i}\n')
    hyp = tgp.make_text(test[i])
    refs = [l['text'] for l in test[i].lexes]
    for t in test[i].triples:
        print(t)
    print(f'\n{hyp}\n')
    for ref in refs:
        print(f'{ref}')
