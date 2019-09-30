# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        extract_refs,
        load_dataset,
        Entry,
        make_template_lm_texts,
        extract_templates,
        preprocess_so
)
import os
import pickle
from collections import defaultdict, Counter, namedtuple
from reg import REGer
from more_itertools import flatten
import subprocess
from random import shuffle
from template_based import JustJoinTemplate
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from random import randint


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')
REFERRER_COUNTER_FILENAME = 'referrer_counter_{}'
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')

LM = namedtuple('LM', ['score'])
RANDOM_LM = LM(lambda t, bos, eos: randint(0, 100000))


# Referring Expression Generation
def load_referrer(dataset_names, referrer_name):

    if referrer_name == 'counter':
        name_db, pronoun_db = load_referrer_counters(dataset_names)

        reger = REGer(name_db, pronoun_db)

        return reger.refer

    if referrer_name == 'inv_counter':
        name_db, pronoun_db = load_referrer_counters(dataset_names)

        reger = REGer(name_db,
                      pronoun_db,
                      name_db_position=-1,
                      pronoun_db_position=-1)

        return reger.refer

    if referrer_name == 'preprocess_so':
        return referrer_preprocess_so


def referrer_preprocess_so(s, ctx):

    return preprocess_so(s)


def load_referrer_counters(dataset_names):

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

    return name_db, pronoun_db


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

    if lm_name == 'random':

        return RANDOM_LM

    if lm_name == 'lower':

        import kenlm

        lm_filename = 'tems_lm_model_lower_{}_{}.arpa'\
            .format(n, '_'.join(sorted(dataset_names)))
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        return kenlm.Model(lm_filepath)

    if lm_name == 'inv_lower':

        import kenlm

        lm_filename = 'tems_lm_model_lower_{}_{}.arpa'\
            .format(n, '_'.join(sorted(dataset_names)))
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        lm = kenlm.Model(lm_filepath)

        def inv_score(t, bos, eos):
            return -1*lm.score(t, bos=bos, eos=eos)

        return LM(inv_score)


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

    if lm_name == 'random':

        return RANDOM_LM

    if lm_name == 'lower':

        import kenlm

        lm_filename = 'txs_lm_model_lower_{}_{}.arpa'\
            .format(n, '_'.join(sorted(dataset_names)))
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        return kenlm.Model(lm_filepath)

    if lm_name == 'inv_lower':

        import kenlm

        lm_filename = 'txs_lm_model_lower_{}_{}.arpa'\
            .format(n, '_'.join(sorted(dataset_names)))
        lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

        lm = kenlm.Model(lm_filepath)

        def inv_score(t, bos, eos):
            return -1*lm.score(t, bos=bos, eos=eos)

        return LM(inv_score)


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


def random_dp_scorer(dps, n_triples):

    return get_random_scores(len(dps))


def load_discourse_planning(dataset_names, dp_name):

    if dp_name == 'random':
        return random_dp_scorer
    if dp_name == 'ltr_lasso':
        return ltr_lasso_dp_scorer(dataset_names)
    if dp_name == 'inv_ltr_lasso':
        scorer = ltr_lasso_dp_scorer(dataset_names)

        def inv_scorer(*args):
            scores = scorer(*args)
            return [-1*x for x in scores]

        return inv_scorer


def ltr_lasso_dp_scorer(dataset_names):

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import Lasso
    import numpy as np

    models = {}

    dataset_names_id = '_'.join(sorted(dataset_names))
    sa_extractor_filename = 'dp_extractor_{}'.format(dataset_names_id)
    sa_extractor_filepath = os.path.join(PRETRAINED_DIR, sa_extractor_filename)

    with open(sa_extractor_filepath, 'rb') as f:
        fe = pickle.load(f)

    for k in range(2, 8):

        sa_data_filename = 'dp_data_{}_{}.npy'.format(dataset_names_id, k)
        sa_data_filepath = os.path.join(PRETRAINED_DIR, sa_data_filename)

        data = np.load(sa_data_filepath)

        X = data[:, :-1]
        y = data[:, -1]

        pipe = Pipeline([
            ('mms',  MinMaxScaler()),
            ('clf', Lasso(alpha=0.0001))
        ])

        pipe.fit(X, y)

        models[k] = pipe

    scorer = get_scorer(models, fe)

    return scorer


# Sentence Aggregation
def random_sa_scorer(sas, n_triples):

    rs = get_random_scores(len(sas))

    ix_1_triple_1_sen = [i for i, sa in enumerate(sas)
                         if len(sa) == n_triples]

    rs[ix_1_triple_1_sen[0]] = 10e5

    return rs


def ltr_lasso_sa_raw_scorer(dataset_names):

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import Lasso
    import numpy as np

    models = {}

    dataset_names_id = '_'.join(sorted(dataset_names))
    sa_extractor_filename = 'sa_extractor_{}'.format(dataset_names_id)
    sa_extractor_filepath = os.path.join(PRETRAINED_DIR, sa_extractor_filename)

    with open(sa_extractor_filepath, 'rb') as f:
        fe = pickle.load(f)

    for k in range(2, 8):

        sa_data_filename = 'sa_data_{}_{}.npy'.format(dataset_names_id, k)
        sa_data_filepath = os.path.join(PRETRAINED_DIR, sa_data_filename)

        data = np.load(sa_data_filepath)

        X = data[:, :-1]
        y = data[:, -1]

        pipe = Pipeline([
            ('mms',  MinMaxScaler()),
            ('clf', Lasso(alpha=0.0001))
        ])

        pipe.fit(X, y)

        models[k] = pipe

    scorer = get_scorer(models, fe)

    return scorer


def ltr_lasso_sa_scorer(dataset_names):

    scorer = ltr_lasso_sa_raw_scorer(dataset_names)

    def score_one_sentence_per_triple_best(os, n_triples):

        ix_one_sen_per_triple = [i for i in range(len(os))
                                 if len(os[i]) == n_triples][0]
        scores = scorer(os, n_triples)

        scores[ix_one_sen_per_triple] = max(scores) + 1

        return scores

    return score_one_sentence_per_triple_best


def inv_ltr_lasso_sa_scorer(dataset_names):

    scorer = ltr_lasso_sa_raw_scorer(dataset_names)

    def score_one_sentence_per_triple_best(os, n_triples):

        ix_one_sen_per_triple = [i for i in range(len(os))
                                 if len(os[i]) == n_triples][0]
        scores = scorer(os, n_triples)
        scores = [-1*x for x in scores]

        scores[ix_one_sen_per_triple] = max(scores) + 1

        return scores

    return score_one_sentence_per_triple_best


def load_sentence_aggregation(dataset_names, sa_name):

    if sa_name == 'random':
        return random_sa_scorer
    if sa_name == 'ltr_lasso':
        return ltr_lasso_sa_scorer(dataset_names)
    if sa_name == 'inv_ltr_lasso':
        return inv_ltr_lasso_sa_scorer(dataset_names)


# Template Fallback
def load_template_fallback(dataset_names, fallback_name):

    if fallback_name == 'jjt':
        return JustJoinTemplate()


# Preprocessing
def load_preprocessing(preprocessing_name):

    if preprocessing_name == 'lower':
        return lambda t: t.lower()
