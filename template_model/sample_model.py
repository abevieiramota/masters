# -*- coding: utf-8 -*-
from plain_experimento import make_model
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from reading_thiagos_templates import Entry, load_shared_task_test, load_test

params = {
        'tems_lm_name': 'lower',
        'txs_lm_name': 'lower',
        'tems_lm_n': 3,
        'tems_lm_bos': False,
        'tems_lm_eos': False,
        'tems_lm_preprocess_input': 'lower',
        'txs_lm_preprocess_input': 'lower',
        'txs_lm_n': 6,
        'txs_lm_bos': False,
        'txs_lm_eos': False,
        'dp_scorer': 'markov_n=2',
        'sa_scorer': 'ltr_lasso',
        'max_tems': 5,
        'max_refs': 5,
        'fallback_template': 'jjt',
        'referrer': 'abe',
        'referrer_lm_n': 3,
        'max_texts': 10,
        'lp_a': 0,
        'lp_n': 0
}

tgp = make_model(params, ('train', 'dev'))

test = load_shared_task_test()
#test = load_test()
