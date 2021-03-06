# -*- coding: utf-8 -*-
from plain_experimento import make_model
from reading_thiagos_templates import Entry, load_shared_task_test, load_test
import logging 
import sys

params = {
        'tems_lm_name': 'markov',
        'txs_lm_name': 'markov',
        'tems_lm_n': 3,
        'txs_lm_n': 5,
        'dp_scorer': 'markov',
        'dp_lm_n': 4,
        'sa_scorer': 'markov',
        'sa_lm_n': 5,
        'max_dp': 10, 
        'max_sa': 10,
        'max_tems': 10,
        'max_refs': 10,
        'fallback_template': 'jjt',
        'referrer': 'abe',
        'referrer_lm_n': 3,
        'lp_txs_a': 0,
        'lp_txs_n': 0,
        'lp_tems_a': 0,
        'lp_tems_n': 0
}

tgp = make_model(params, ('train', 'dev'))

test = load_shared_task_test()

ix = int(sys.argv[1]) if len(sys.argv) > 1 else 0

log_level = sys.argv[2] if len(sys.argv) > 2 else 'DEBUG'

logging.basicConfig(level=log_level)

e = test[ix]

logging.getLogger('sample_model.py').info(''.join('\n<{} - {} - {}>'.format(*t) for t in e.triples))

print(tgp.make_text(e))
