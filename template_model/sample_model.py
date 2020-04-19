# -*- coding: utf-8 -*-
from plain_experimento import make_model
from reading_thiagos_templates import Entry, load_shared_task_test, load_test
import logging 
import sys

logging.basicConfig(level='DEBUG')

params = {
        'tems_lm_name': 'lower',
        'txs_lm_name': 'lower',
        'tems_lm_n': 4,
        'txs_lm_n': 4,
        'dp_scorer': 'markov',
        'dp_scorer_n': 4,
        'sa_scorer': 'markov',
        'sa_scorer_n': 3,
        'max_dp': 1, 
        'max_sa': 1,
        'max_tems': 2,
        'max_refs': 2,
        'fallback_template': 'jjt',
        'referrer': 'abe',
        'referrer_lm_n': 4,
        'lp_a': 1,
        'lp_n': 0
}

tgp = make_model(params, ('train', 'dev'))

test = load_shared_task_test()

ix = int(sys.argv[1]) if len(sys.argv) > 1 else 0

print(tgp.make_text(test[ix]))
