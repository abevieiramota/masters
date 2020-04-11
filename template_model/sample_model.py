# -*- coding: utf-8 -*-
from plain_experimento import make_model
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from reading_thiagos_templates import Entry, load_shared_task_test, load_test

params = {
        'tems_lm_name': 'lower',
        'txs_lm_name': 'lower',
        'tems_lm_n': 3,
        'txs_lm_n': 3,
        'dp_scorer': 'markov',
        'dp_scorer_n': 3,
        'sa_scorer': 'markov',
        'sa_scorer_n': 3,
        'max_dp': 2, 
        'max_sa': 2,
        'max_tems': 2,
        'max_refs': 2,
        'fallback_template': 'jjt',
        'referrer': 'abe',
        'referrer_lm_n': 3,
        'lp_a': 0,
        'lp_n': 0
}

tgp = make_model(params, ('train', 'dev'))

test = load_shared_task_test()
