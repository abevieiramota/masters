# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        extract_refs,
        load_dataset,
        Entry,
        make_template_lm_texts,
        extract_templates
)
import os
import pickle
from collections import defaultdict, Counter
from reg import REGer
from more_itertools import flatten
import subprocess
from random import shuffle
from template_based import JustJoinTemplate


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')
REFERRER_COUNTER_FILENAME = 'referrer_counter_{}'
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')


# Referring Expression Generation
def load_referrer(dataset_names, referrer_name):

    if referrer_name == 'counter':

        return load_counter_referrer(dataset_names)


def load_counter_referrer(dataset_names):

    name_dbs, pronoun_dbs = [], []
    for dataset_name in dataset_names:
        name_db, pronoun_db = load_name_pronoun_db(dataset_name)
        name_dbs.append(name_db)
        pronoun_dbs.append(pronoun_db)
    # union of name_db s
    name_db = defaultdict(lambda: Counter())
    for name_db_ in name_dbs:
        for k, v in name_db_.items():
            name_db[k] += v
    # union of pronoun_db s
    pronoun_db = defaultdict(lambda: Counter())
    for pronoun_db_ in pronoun_db:
        for k, v in pronoun_db_.items():
            pronoun_db[k] += v

    reger = REGer(name_db, pronoun_db)

    return reger.refer


def load_name_pronoun_db(dataset_name):

    referrer_db_filepath = os.path.join(PRETRAINED_DIR,
                                        REFERRER_COUNTER_FILENAME.format(
                                                dataset_name))
    with open(referrer_db_filepath, 'rb') as f:
        data = pickle.load(f)

    return data['name_db'], data['pronoun_db']


def make_pretrained_name_pronoun_db(dataset_name):

    dataset = load_dataset(dataset_name)

    referrer_db_filepath = os.path.join(PRETRAINED_DIR,
                                        REFERRER_COUNTER_FILENAME.format(
                                                dataset_name))

    name_db, pronoun_db = extract_refs(dataset)

    with open(referrer_db_filepath, 'wb') as f:
        data = {'name_db': name_db,
                'pronoun_db': pronoun_db}
        pickle.dump(data, f)


# Template Selection Language Models
def load_template_selection_lm(dataset_names, n, lm_name):

    import kenlm

    lm_filename = 'tems_lm_model_{}_{}_{}.arpa'\
        .format(lm_name, n, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    return kenlm.Model(lm_filepath)


def make_template_selection_lm(dataset_names,
                               n,
                               lm_name,
                               preprocessing):

    texts_filename = 'tems_lm_texts_{}_{}.txt'\
        .format(lm_name, '_'.join(sorted(dataset_names)))
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)
    if not os.path.isfile(texts_filepath):
        dataset = list(flatten(load_dataset(ds_name)
                               for ds_name in dataset_names))
        e_t, _ = extract_templates(dataset)
        tems_lm_texts = make_template_lm_texts(e_t)
        tems_lm_texts = [preprocessing(t) for t in tems_lm_texts]

        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in tems_lm_texts:
                f.write(f'{t}\n')
    with open(texts_filepath, 'rb') as f:
        tems_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                         stdout=subprocess.PIPE,
                                         input=f.read())

    lm_filename = 'tems_lm_model_{}_{}_{}.arpa'\
        .format(lm_name, n, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)
    with open(lm_filepath, 'wb') as f:
        f.write(tems_lm_process.stdout)


# Text Selection Language Models
def load_text_selection_lm(dataset_names, n, lm_name):

    import kenlm

    lm_filename = 'txs_lm_model_{}_{}_{}.arpa'\
        .format(lm_name, n, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    return kenlm.Model(lm_filepath)


def make_text_selection_lm(dataset_names,
                           n,
                           lm_name,
                           preprocessing):

    texts_filename = 'txs_lm_texts_{}_{}.txt'\
        .format(lm_name, '_'.join(sorted(dataset_names)))
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)
    if not os.path.isfile(texts_filepath):
        dataset = list(flatten(load_dataset(ds_name)
                               for ds_name in dataset_names))
        txs_lm_texts = [l['text']
                        for e in dataset
                        for l in e.lexes
                        if l['comment'] == 'good']
        txs_lm_texts = [preprocessing(t) for t in txs_lm_texts]

        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in txs_lm_texts:
                f.write(f'{t}\n')
    with open(texts_filepath, 'rb') as f:
        txs_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = 'txs_lm_model_{}_{}_{}.arpa'\
        .format(lm_name, n, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)
    with open(lm_filepath, 'wb') as f:
        f.write(txs_lm_process.stdout)


# Template DB
def load_template_db(dataset_names):

    template_db_filename = 'template_db_{}'.format(
            '_'.join(sorted(dataset_names)))
    template_db_filepath = os.path.join(PRETRAINED_DIR, template_db_filename)

    with open(template_db_filepath, 'rb') as f:
        template_db = pickle.load(f)

    return template_db


def make_template_db(dataset_names):

    dataset = list(flatten(load_dataset(ds_name)
                           for ds_name in dataset_names))
    e_t, _ = extract_templates(dataset)

    template_db = defaultdict(set)

    for e, lexes_templates in e_t:

        for l, ts in lexes_templates:

            for t in ts:

                template_db[(e.category, t.template_triples)].add(t)

    template_db = dict(template_db)

    template_db_filename = 'template_db_{}'.format(
            '_'.join(sorted(dataset_names)))
    template_db_filepath = os.path.join(PRETRAINED_DIR, template_db_filename)

    with open(template_db_filepath, 'wb') as f:
        pickle.dump(template_db, f)


# Discourse Planning
def get_random_scores(n):

    rs = list(range(n))
    # TODO: review how random is being handled
    shuffle(rs)

    return rs


def random_dp_scorer(dps, n_triples):

    return get_random_scores(len(dps))


def load_discourse_planning(dataset_names, dp_name):

    if dp_name == 'random':
        return random_dp_scorer


# Sentence Aggregation
def random_sa_scorer(sas, n_triples):

    rs = get_random_scores(len(sas))

    ix_1_triple_1_sen = [i for i, sa in enumerate(sas)
                         if len(sa) == n_triples]

    rs[ix_1_triple_1_sen[0]] = 10e5

    return rs


def load_sentence_aggregation(dataset_names, sa_name):

    if sa_name == 'random':
        return random_sa_scorer


# Template Fallback
def load_template_fallback(dataset_names, fallback_name):

    if fallback_name == 'jjt':
        return JustJoinTemplate()
