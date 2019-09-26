# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        Entry
)
import os
from more_itertools import flatten
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
        load_template_fallback
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMS_LM_TEXT_FILENAME = 'tems_lm_text.txt'
TEMS_LM_MODEL_FILENAME = 'tems_lm.arpa'
TXS_LM_TEXT_FILENAME = 'txs_lm_text.txt'
TXS_LM_MODEL_FILENAME = 'txs_lm.arpa'
TEMPLATE_DB_FILENAME = 'template_db'
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')

params = {
        'train_set': ['train'],
        'test_set': ['dev'],
        'model_name': 'abe-random',
        'tems_lm_name': 'lower',
        'tems_lm_n': 6,
        'tems_lm_bos': False,
        'tems_lm_eos': False,
        'tems_lm_preprocess_input': lambda t: t.lower(),
        'txs_lm_preprocess_input': lambda t: t.lower(),
        'txs_lm_name': 'lower',
        'txs_lm_n': 6,
        'txs_lm_bos': False,
        'txs_lm_eos': False,
        'dp_scorer': 'random',
        'sa_scorer': 'random',
        'max_dp': 2,
        'max_sa': 3,
        'max_tems': 5,
        'fallback_template': 'jjt',
        'referrer': 'counter'
}

train = list(flatten(load_dataset(ds) for ds in params['train_set']))
test = list(flatten(load_dataset(ds) for ds in params['test_set']))
model_name = params['model_name']
tems_lm_bos = params['tems_lm_bos']
tems_lm_eos = params['tems_lm_eos']
txs_lm_n = params['txs_lm_n']
txs_lm_bos = params['txs_lm_bos']
txs_lm_eos = params['txs_lm_eos']
max_sa = params['max_sa']
max_dp = params['max_dp']
max_tems = params['max_tems']
tems_lm_preprocess_input = params['tems_lm_preprocess_input']
txs_lm_preprocess_input = params['txs_lm_preprocess_input']

# 1. Grid Search
# 1.1. Language Models
# 1.1.1 Template Selection Language Model
tems_lm = load_template_selection_lm(params['train_set'],
                                     params['tems_lm_n'],
                                     params['tems_lm_name'])
# 1.1.2 Text Selection Language Model
txs_lm = load_text_selection_lm(params['train_set'],
                                params['txs_lm_n'],
                                params['txs_lm_name'])
# 1.2 Template database
template_db = load_template_db(params['train_set'])
# 1.3 Referring Expression Generation
# -> extraction and dump is in pretrained_models.py
# 1.3.0 References extraction
# 1.3.1 Save them
# 1.3.2 Referrer
referrer = load_referrer(params['train_set'], params['referrer'])
# 1.4 Discourse Planning
dp_scorer = load_discourse_planning(params['train_set'],
                                    params['dp_scorer'])
# 1.5 Sentence Aggregation
sa_scorer = load_sentence_aggregation(params['train_set'],
                                      params['sa_scorer'])
# 1.6 Template Fallback
fallback_template = load_template_fallback(params['train_set'],
                                           params['fallback_template'])


# ---------------- Model
tgp = TextGenerationPipeline(
        template_db,
        tems_lm,
        tems_lm_bos,
        tems_lm_eos,
        tems_lm_preprocess_input,
        txs_lm,
        txs_lm_bos,
        txs_lm_eos,
        txs_lm_preprocess_input,
        dp_scorer,
        sa_scorer,
        max_dp,
        max_sa,
        max_tems,
        fallback_template,
        referrer)
