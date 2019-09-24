# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        make_template_lm_texts,
        extract_templates,
        make_template_db,
        extract_refs,
        Entry
)
import os
import subprocess
from more_itertools import flatten
from plain_experimento import (
        TextGenerationPipeline,
        random_dp_scorer,
        random_sa_scorer
)
import pickle
import kenlm
from template_based import JustJoinTemplate
from reg import REGer
from pretrained_models import load_name_pronoun_db
from collections import defaultdict, Counter


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
        'tems_train_preprocess_text': lambda t: t.lower(),
        'txs_train_preprocess_text': lambda t: t.lower(),
        'tems_lm_n': 6,
        'tems_lm_bos': False,
        'tems_lm_eos': False,
        'tems_lm_preprocess_input': lambda t: t.lower(),
        'txs_lm_preprocess_input': lambda t: t.lower(),
        'txs_lm_bos': False,
        'txs_lm_eos': False,
        'dp_scorer': 'random',
        'sa_scorer': 'random',
        'max_dp': 2,
        'max_sa': 3,
        'max_tems': 5,
        'fallback_template': 'jjt',
        'referrer': 'pretrained_counter'
}

train = flatten(load_dataset(ds) for ds in params['train_set'])
test = flatten(load_dataset(ds) for ds in params['test_set'])
tems_train_preprocess_text = params['tems_train_preprocess_text']
txs_train_preprocess_text = params['txs_train_preprocess_text']
model_name = params['model_name']
tems_lm_n = str(params['tems_lm_n'])
tems_lm_bos = params['tems_lm_bos']
tems_lm_eos = params['tems_lm_eos']
txs_lm_bos = params['txs_lm_bos']
txs_lm_eos = params['txs_lm_eos']
if params['dp_scorer'] == 'random':
    dp_scorer = random_dp_scorer
if params['sa_scorer'] == 'random':
    sa_scorer = random_sa_scorer
max_sa = params['max_sa']
max_dp = params['max_dp']
max_tems = params['max_tems']
if params['fallback_template'] == 'jjt':
    fallback_template = JustJoinTemplate()
tems_lm_preprocess_input = params['tems_lm_preprocess_input']
txs_lm_preprocess_input = params['txs_lm_preprocess_input']
model_dir = os.path.join(BASE_DIR, f'../data/models/{model_name}')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
tems_lm_text_filepath = os.path.join(model_dir, TEMS_LM_TEXT_FILENAME)
txs_lm_text_filepath = os.path.join(model_dir, TXS_LM_TEXT_FILENAME)
tems_lm_model_filepath = os.path.join(model_dir, TEMS_LM_MODEL_FILENAME)
txs_lm_model_filepath = os.path.join(model_dir, TXS_LM_MODEL_FILENAME)
template_db_filepath = os.path.join(model_dir, TEMPLATE_DB_FILENAME)

# 1. Grid Search

# 1.0 Template extraction
# ! importante -> nem todo texto de referência vira template
# !     as vezes o processo falha -> é importante analisar isso!

e_t, errors = extract_templates(train)

# 1.1. Language Models

# 1.1.1 Template Selection Language Model
# 1.1.1.0 Texts
tems_lm_texts = make_template_lm_texts(e_t)
# 1.1.1.1 Preprocessing
tems_lm_texts = [tems_train_preprocess_text(t) for t in tems_lm_texts]
# 1.1.1.2 Creating file
with open(tems_lm_text_filepath, 'w', encoding='utf-8') as f:
    for t in tems_lm_texts:
        f.write(f'{t}\n')
# 1.1.1.3 Creating the language model
with open(tems_lm_text_filepath, 'rb') as f:
    tems_lm_process = subprocess.run([KENLM, '-o', tems_lm_n],
                                     stdout=subprocess.PIPE,
                                     input=f.read())
# 1.1.1.4 Save it
with open(tems_lm_model_filepath, 'wb') as f:
    f.write(tems_lm_process.stdout)
# 1.1.1.5 Load it
tems_lm = kenlm.Model(tems_lm_model_filepath)

# 1.1.2 Text Selection Language Model
# 1.1.2.0 Texts
txs_lm_texts = [l['text'] for e in train for l in e.lexes
                if l['comment'] == 'good']
# 1.1.2.1 Preprocessing
txs_lm_texts = [txs_train_preprocess_text(t) for t in txs_lm_texts]
# 1.1.2.2 Creating file
with open(txs_lm_text_filepath, 'w', encoding='utf-8') as f:
    for t in txs_lm_texts:
        f.write(f'{t}\n')
# 1.1.2.3 Creating the language model
with open(txs_lm_text_filepath, 'rb') as f:
    txs_lm_process = subprocess.run([KENLM, '-o', tems_lm_n],
                                    stdout=subprocess.PIPE,
                                    input=f.read())
# 1.1.2.4 Save it
with open(txs_lm_model_filepath, 'wb') as f:
    f.write(txs_lm_process.stdout)
# 1.1.2.5 Load it
txs_lm = kenlm.Model(txs_lm_model_filepath)

# 1.2 Template database
# 1.2.0 Make template db data
template_db = make_template_db(e_t)
# 1.2.1 Save it
with open(template_db_filepath, 'wb') as f:
    pickle.dump(template_db, f)

# 1.3 Referring Expression Generation
# -> extraction and dump is in pretrained_models.py
# 1.3.0 References extraction
# 1.3.1 Save them
# 1.3.2 Referrer
name_dbs, pronoun_dbs = [], []
for dataset_name in params['train_set']:
    name_db, pronoun_db = load_name_pronoun_db(dataset_name,
                                               params['referrer'])
    name_dbs.append(name_db)
    pronoun_dbs.append(pronoun_db)
name_db = defaultdict(lambda: Counter())
for name_db_ in name_dbs:
    for k, v in name_db_.items():
        name_db[k] += v

pronoun_db = defaultdict(lambda: Counter())
for pronoun_db_ in pronoun_db:
    for k, v in pronoun_db_.items():
        pronoun_db[k] += v

reger = REGer(name_db, pronoun_db)
referrer = reger.refer


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
