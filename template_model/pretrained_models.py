# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        load_datasets,
        Entry,
        extract_templates,
        get_lexicalizations
)
import os
import pickle
from collections import defaultdict, Counter, namedtuple
from reg import EmptyREGer, FirstSecondREG, PreprocessREG
import subprocess
from random import shuffle
from template_based import JustJoinTemplate, TemplateDatabase
from random import randint
from functools import lru_cache
from testing_make_reg_lm_db import extract_text_reg_lm
import re
import kenlm
from preprocessing import *
from more_itertools import flatten



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')

LM = namedtuple('LM', ['score'])
RANDOM_LM = LM(lambda t: randint(0, 100000))


# Referring Expression Generation
@lru_cache(maxsize=10)
def load_referrer(db_names, referrer_name, reg_lm_n=None):

    if referrer_name == 'preprocess_so':
        return PreprocessREG()

    if referrer_name == 'empty':
        return EmptyREGer()

    if referrer_name == 'abe':
        ref_db = load_abe_referrer_counters(db_names)
        ref_lm = load_reg_lm(db_names, reg_lm_n)

        reger = FirstSecondREG(ref_db, ref_lm)

        return reger


def load_abe_referrer_counters(db_names):

    ref_db = defaultdict(lambda: defaultdict(set))
    for db_name in db_names:
        filename = f'abe_referrer_counter_{db_name}'
        referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

        with open(referrer_db_filepath, 'rb') as f:
            data = pickle.load(f)

        for type_, entity_c in data.items():
            for entity, c in entity_c.items():
                ref_db[type_][entity].update(c)

    return ref_db


def load_reg_lm(db_names, n):

    db_names_id = '_'.join(sorted(db_names))

    lm_filename = f'reg_lm_model_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    return kenlm.Model(lm_filepath)


def make_reg_lm(db_names, n):

    db_names_id = '_'.join(sorted(db_names))

    texts_filename = f'reg_lm_texts_{db_names_id}.txt'
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)

    if not os.path.isfile(texts_filepath):
        dataset = load_datasets(db_names)
        texts = []

        for e in dataset:

            good_lexes = [l for l in e.lexes if l['comment'] == 'good']

            for l in good_lexes:

                t = extract_text_reg_lm(l['text'], l['template'])
                if t:
                    texts.append(t)

        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(f'{t}\n')

    with open(texts_filepath, 'rb') as f:
        reg_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = f'reg_lm_model_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    with open(lm_filepath, 'wb') as f:
        f.write(reg_lm_process.stdout)


def make_pretrained_abe_ref_dbs(db_name):

    dataset = load_dataset(db_name)

    ref_db = defaultdict(lambda: defaultdict(set))

    lex_keys = set()

    for e in dataset:

        good_lexes = (l for l in e.lexes if l['comment'] == 'good' and e.entity_map)

        for l in good_lexes:

            lexicals = get_lexicalizations(l['text'], l['template'], e.entity_map)

            for lex_key, lex_values in lexicals.items():
                lex_keys.add(lex_key)

                for i, lex_value in enumerate(lex_values):

                    lex_value = normalize_text(lex_value)

                    if i == 0:
                        ref_db['1st'][lex_key].add(lex_value)
                    else:
                        ref_db['2nd'][lex_key].add(lex_value)


    filename = f'abe_referrer_counter_{db_name}'
    referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

    ref_db = {k: dict(v) for k, v in ref_db.items()}

    with open(referrer_db_filepath, 'wb') as f:
        pickle.dump(ref_db, f)


# Template Selection Models

@lru_cache(maxsize=10)
def load_template_selection_lm(db_names, n, lm_name):

    if lm_name == 'random':
        return RANDOM_LM

    if lm_name == 'markov':

        db_names_id = '_'.join(sorted(db_names))

        lm_filename = f'tems_lm_model_markov_{n}_{db_names_id}.arpa'
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        return kenlm.Model(lm_filepath)

    if lm_name == 'inv_markov':

        lm = load_template_selection_lm(db_names, n, 'markov')

        def inv_score(t):
            return -1*lm.score(t)

        return LM(inv_score)


def make_template_lm_texts(entries_templates):

    template_lm_texts = []

    for _, lexes_templates in entries_templates:
        for l, ts in lexes_templates:
            for tem, triples in zip(ts, l['sorted_triples']):

                if tem:                
                    # map de slot -> subject/object
                    aligned_data = tem.align(triples)
                    # cria lista de referências, onde referência é
                    #    identificador do subject/object
                    refs = [text_to_id(aligned_data[slot_name]) for slot_name, _ in tem.slots]
                    text = tem.fill(refs)

                    template_lm_texts.append(text)

    return template_lm_texts


def make_template_selection_lm(db_names, n, lm_name):

    db_names_id = '_'.join(sorted(db_names))

    texts_filename = f'tems_lm_texts_{lm_name}_{db_names_id}.txt'
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)

    if not os.path.isfile(texts_filepath):
        dataset = load_datasets(db_names)

        e_t = extract_templates(dataset)
        tems_lm_texts = make_template_lm_texts(e_t)

        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in tems_lm_texts:
                f.write(f'{t}\n')

    with open(texts_filepath, 'rb') as f:
        tems_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                         stdout=subprocess.PIPE,
                                         input=f.read())

    lm_filename = f'tems_lm_model_{lm_name}_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)
    with open(lm_filepath, 'wb') as f:
        f.write(tems_lm_process.stdout)


# Text Selection Language Models
@lru_cache(maxsize=10)
def load_text_selection_lm(db_names, n, lm_name):

    db_names_id = '_'.join(sorted(db_names))

    if lm_name == 'random':
        return RANDOM_LM

    if lm_name == 'markov':

        lm_filename = f'txs_lm_model_markov_{n}_{db_names_id}.arpa'
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        return kenlm.Model(lm_filepath)

    if lm_name == 'inv_markov':

        lm = load_text_selection_lm(db_names, n, lm_name)

        def inv_score(t):
            return -1*lm.score(t)

        return LM(inv_score)


def make_text_selection_lm(db_names, n, lm_name):

    db_names_id = '_'.join(sorted(db_names))

    texts_filename = f'txs_lm_texts_{db_names_id}.txt'
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)

    if not os.path.isfile(texts_filepath):
        dataset = load_datasets(db_names)
        txs_lm_texts = [normalize_thiagos_template(l['text'].lower())
                        for e in dataset
                        for l in e.lexes
                        if l['comment'] == 'good']
        txs_lm_texts = [normalize_text(t) for t in txs_lm_texts]

        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in txs_lm_texts:
                f.write(f'{t}\n')
    with open(texts_filepath, 'rb') as f:
        txs_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = f'txs_lm_model_{lm_name}_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    with open(lm_filepath, 'wb') as f:
        f.write(txs_lm_process.stdout)


# Template DB
@lru_cache(maxsize=10)
def load_template_db(db_names, fallback_template=None, ns=None):

    db_names_id = '_'.join(sorted(db_names))

    template_db_filename = f'template_db_{db_names_id}'
    template_db_filepath = os.path.join(PRETRAINED_DIR, template_db_filename)

    with open(template_db_filepath, 'rb') as f:
        template_db = pickle.load(f)

    if ns is None:
        return TemplateDatabase(template_db, fallback_template)
    else:
        tdb = {k: v for k, v in template_db.items()
               if len(k[1]) in ns}
        return TemplateDatabase(tdb, fallback_template)


def make_template_db(db_names):

    db_names_id = '_'.join(sorted(db_names))
    dataset = load_datasets(db_names)
    e_t = extract_templates(dataset)

    template_db = defaultdict(set)

    for e, lexes_templates in e_t:

        for _, ts in lexes_templates:

            for t in [t for t in ts if t]:

                template_db[t.template_triples].add(t)

    template_db = dict(template_db)

    template_db_filename = f'template_db_{db_names_id}'
    template_db_filepath = os.path.join(PRETRAINED_DIR, template_db_filename)

    with open(template_db_filepath, 'wb') as f:
        pickle.dump(template_db, f)


# Common functions
def get_scorer(models, fe):

    def scorer(os, n_triples):

        if n_triples == 1:
            return [-1]

        data = fe[n_triples].transform(os)

        scores = models[n_triples].predict(data)

        return scores

    return scorer


# Discourse Planning
def get_random_scores(n):

    rs = list(range(n))
    # TODO: review how random is being handled
    shuffle(rs)

    return rs


def random_dp_scorer(dps):

    return get_random_scores(len(dps))


def preprocess_to_dp_model(triples):

    pred_text = ' '.join(text_to_id(t.predicate) for t in triples)

    return pred_text


@lru_cache(maxsize=10)
def load_discourse_planning(db_names, dp_name, n=None):

    if dp_name == 'random':
        return random_dp_scorer

    if dp_name == 'markov':

        db_names_id = '_'.join(sorted(db_names))

        lm_filename = f'dp_lm_model_{n}_{db_names_id}.arpa'
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        model = kenlm.Model(lm_filepath)

        def scorer(triples_list):

            scores = []
            for triples in triples_list:
                pred_text = preprocess_to_dp_model(triples)
                score = model.score(pred_text)
                scores.append(score)

            return scores
        return scorer


def extract_orders(e):

    for l in e.lexes:
        if l['comment'] == 'good' and l['sorted_triples']:

            order = tuple(flatten(l['sorted_triples']))
            if len(order) == len(e.triples):
                yield order


def make_dp_lm(db_names, n=2):
    # constrói o modelo de scoring de discourse plans

    # lê as entradas dos bancos de nomes em db_names
    dataset = load_datasets(db_names)
    db_names_id = '_'.join(sorted(db_names))

    orders = []

    # para cada entrada, com len(triples) >= 2, extrai as ordens com que as triplas
    #    foram verbalizadas 
    for e in dataset:
        if len(e.triples) >= 2:
            order = extract_orders(e)
            orders.extend(order)

    dp_lm_texts_filename = f'txs_dp_texts_{db_names_id}.txt'
    dp_lm_texts_filepath = os.path.join(PRETRAINED_DIR, dp_lm_texts_filename)

    with open(dp_lm_texts_filepath, 'w') as f:
        for order in orders:
            preprocessed_order = preprocess_to_dp_model(order)
            f.write(f'{preprocessed_order}\n')

    with open(dp_lm_texts_filepath, 'rb') as f:
        txs_lm_process = subprocess.run([KENLM, '-o', str(n)],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = f'dp_lm_model_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)
    with open(lm_filepath, 'wb') as f:
        f.write(txs_lm_process.stdout)


# Sentence Aggregation

def preprocess_to_sa_model(triples_partitioned):

    pred_text = ' | '.join(' _ '.join(text_to_id(t.predicate) for t in part) 
                           for part in triples_partitioned)

    return pred_text


@lru_cache(maxsize=10)
def load_sentence_aggregation(db_names, sa_name, n=None):

    if sa_name == 'random':
        def random_sa_scorer(sas):
            return get_random_scores(len(sas))
        return random_sa_scorer

    if sa_name == 'markov':

        db_names_id = '_'.join(sorted(db_names))

        lm_filename = f'sa_lm_model_{n}_{db_names_id}.arpa'
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        model = kenlm.Model(lm_filepath)

        def scorer(sas):
            # sas é uma lista de triplas particionadas
            #    por triplas particionadas, quero dizer uma lista de listas de triplas
            #    ex: [[t1, t2], [t3]]

            scores = []
            for sa in sas:
                pred_text = preprocess_to_sa_model(sa)
                score = model.score(pred_text)
                scores.append(score)

            return scores
        return scorer


def extract_partitionings(e):

    for l in e.lexes:
        if l['comment'] == 'good' and l['sorted_triples'] and len(e.triples) == sum(len(part) for part in l['sorted_triples']):
            yield l['sorted_triples']

    
def make_sa_lm(db_names, n=2):

    dataset = load_datasets(db_names)
    db_names_id = '_'.join(sorted(db_names))

    sas = []

    for e in (e for e in dataset if len(e.triples) >= 2):
        partitionings = extract_partitionings(e)
        sas.extend(partitionings)

    sa_lm_texts_filename= f'txs_sa_texts_{db_names_id}.txt'
    sa_lm_texts_filepath = os.path.join(PRETRAINED_DIR, sa_lm_texts_filename)

    with open(sa_lm_texts_filepath, 'w') as f:
        for sa in sas:
            preprocessed_sa = preprocess_to_sa_model(sa)
            f.write(f'{preprocessed_sa}\n')

    with open(sa_lm_texts_filepath, 'rb') as f:
        #TODO: revisar esse --discount_fallback -> lembro de ter a ver com alguma limitação dos dados utilizados...
        txs_lm_process = subprocess.run([KENLM, '--discount_fallback', '-o', str(n)],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = f'sa_lm_model_{n}_{db_names_id}.arpa'
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)
    with open(lm_filepath, 'wb') as f:
        f.write(txs_lm_process.stdout)


# Template Fallback
def load_template_fallback(dataset_names, fallback_name):

    if fallback_name == 'jjt':
        return JustJoinTemplate
