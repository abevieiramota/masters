# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        extract_thiagos_refs,
        load_dataset,
        Entry,
        make_template_lm_texts,
        extract_templates,
        get_lexicalizations,
        normalize_thiagos_template
)
import os
import pickle
from collections import defaultdict, Counter, namedtuple
from reg import REGer, EmptyREGer, FirstNameOthersPronounREG
from more_itertools import flatten
import subprocess
from random import shuffle
from template_based import JustJoinTemplate
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from random import randint
from util import preprocess_so
from functools import lru_cache
from testing_make_reg_lm_db import extract_text_reg_lm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')
THIAGOS_REFERRER_COUNTER_FILENAME = 'thiagos_referrer_counter_{}'
ABE_REFERRER_COUNTER_FILENAME = 'abe_referrer_counter_{}'
KENLM = os.path.join(BASE_DIR, '../../kenlm/build/bin/lmplz')

LM = namedtuple('LM', ['score'])
RANDOM_LM = LM(lambda t, bos, eos: randint(0, 100000))


# Referring Expression Generation
@lru_cache(maxsize=10)
def load_referrer(dataset_names, referrer_name, max_ref=0):

    if referrer_name == 'thiagos':
        ref_db = load_thiagos_referrer_counters(dataset_names)

        reger = REGer(ref_db)

        return reger

    if referrer_name == 'inv_thiagos':
        ref_db = load_thiagos_referrer_counters(dataset_names)

        reger = REGer(ref_db,
                      name_db_position=-1,
                      pronoun_db_position=-1,
                      description_db_position=-1,
                      demonstrative_db_position=-1)

        return reger

    if referrer_name == 'preprocess_so':
        class REG_:
            def refer(s, slot_pos, slot_type, ctx):
                return preprocess_so(s)
        return REG_()

    if referrer_name == 'random':
        ref_db = load_thiagos_referrer_counters(dataset_names)

        reger = REGer(ref_db, is_random=True, seed=1415)

        return reger

    if referrer_name == 'empty':
        return EmptyREGer()

    if referrer_name == 'abe':
        ref_db = load_abe_referrer_counters(dataset_names)
        ref_lm = load_reg_lm(dataset_names)

        reger = FirstNameOthersPronounREG(ref_db, ref_lm)

        return reger


def load_thiagos_referrer_counters(dataset_names):

    ref_db = defaultdict(lambda: defaultdict(lambda: Counter()))
    for dataset_name in dataset_names:
        data = load_thiagos_ref_dbs(dataset_name)

        for type_, entity_c in data.items():
            for entity, c in entity_c.items():
                ref_db[type_][entity] += c

    return ref_db


def load_thiagos_ref_dbs(dataset_name):

    filename = THIAGOS_REFERRER_COUNTER_FILENAME.format(dataset_name)
    referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

    with open(referrer_db_filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def load_abe_referrer_counters(dataset_names):

    ref_db = defaultdict(lambda: defaultdict(lambda: Counter()))
    for dataset_name in dataset_names:
        data = load_abe_ref_dbs(dataset_name)

        for type_, entity_c in data.items():
            for entity, c in entity_c.items():
                ref_db[type_][entity] += c

    return ref_db


def load_abe_ref_dbs(dataset_name):

    filename = ABE_REFERRER_COUNTER_FILENAME.format(dataset_name)
    referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

    with open(referrer_db_filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def load_reg_lm(dataset_names):

    import kenlm

    lm_filename = 'reg_lm_model_{}_{}.arpa'\
        .format(3, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    return kenlm.Model(lm_filepath)


def make_reg_lm(dataset_names):

    texts_filename = 'reg_lm_texts_{}.txt'\
        .format('_'.join(sorted(dataset_names)))
    texts_filepath = os.path.join(PRETRAINED_DIR, texts_filename)

    if not os.path.isfile(texts_filepath):
        dataset = list(flatten(load_dataset(ds_name)
                               for ds_name in dataset_names))
        texts = []

        for e in dataset:

            good_lexes = [l for l in e.lexes
                          if l['comment'] == 'good']

            for l in good_lexes:

                t = extract_text_reg_lm(l)
                if t:
                    texts.append(t)
        with open(texts_filepath, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(f'{t.lower()}\n')

    with open(texts_filepath, 'rb') as f:
        reg_lm_process = subprocess.run([KENLM, '-o', '3'],
                                        stdout=subprocess.PIPE,
                                        input=f.read())

    lm_filename = 'reg_lm_model_{}_{}.arpa'\
        .format(3, '_'.join(sorted(dataset_names)))
    lm_filepath = os.path.join(PRETRAINED_DIR, lm_filename)

    with open(lm_filepath, 'wb') as f:
        f.write(reg_lm_process.stdout)


def make_pretrained_abe_ref_dbs(dataset_name):

    dataset = load_dataset(dataset_name)

    ref_db = defaultdict(lambda: defaultdict(lambda: Counter()))

    w_errors = []

    for e in dataset:

        good_lexes = [l for l in e.lexes
                      if l['comment'] == 'good' and e.entity_map]

        for l in good_lexes:

            lexicals = get_lexicalizations(l['text'],
                                           l['template'],
                                           e.entity_map)

            if not lexicals:
                w_errors.append((e, l['text'], l['template']))

            for lex_key, lex_values in lexicals.items():
                for i, lex_value in enumerate(lex_values):

                    if i == 0:
                        ref_db['1st'][lex_key][lex_value] += 1
                    else:
                        ref_db['2nd'][lex_key][lex_value] += 1

    filename = ABE_REFERRER_COUNTER_FILENAME.format(dataset_name)
    referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

    ref_db = {k: dict(v) for k, v in ref_db.items()}

    with open(referrer_db_filepath, 'wb') as f:
        pickle.dump(ref_db, f)


def make_pretrained_thiagos_ref_dbs(dataset_name):

    dataset = load_dataset(dataset_name)

    filename = THIAGOS_REFERRER_COUNTER_FILENAME.format(dataset_name)
    referrer_db_filepath = os.path.join(PRETRAINED_DIR, filename)

    ref_db = extract_thiagos_refs(dataset)

    with open(referrer_db_filepath, 'wb') as f:
        pickle.dump(ref_db, f)


# Template Selection Language Models
@lru_cache(maxsize=10)
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
        tems_lm_texts = [normalize_thiagos_template(t) for t in tems_lm_texts]
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
@lru_cache(maxsize=10)
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
        txs_lm_texts = [normalize_thiagos_template(l['text'])
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
@lru_cache(maxsize=10)
def load_template_db(dataset_names, ns=None):

    template_db_filename = 'template_db_{}'.format(
            '_'.join(sorted(dataset_names)))
    template_db_filepath = os.path.join(PRETRAINED_DIR, template_db_filename)

    with open(template_db_filepath, 'rb') as f:
        template_db = pickle.load(f)

    if ns is None:
        return template_db
    else:
        return {k: v for k, v in template_db.items()
                if len(k[1]) in ns}


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


def gold_dp_scorer(dataset_names):

    dev = load_dataset('dev')
    i = 0

    def gold_dp_scorer_(dps, n_triples):

        nonlocal i

        sts = []
        for l in dev[i].lexes:
            st = l['sorted_triples']
            if st:
                sts.append(tuple(flatten(st)))

        scores = []
        for dp in dps:
            if dp in sts:
                scores.append(1)
            else:
                scores.append(0)
        i = i + 1

        return scores

    return gold_dp_scorer_


@lru_cache(maxsize=10)
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
    # ! atualmente só funciona se o target for dev
    # !    e a geração dos textos for na ordem das entries no load_dev()
    if dp_name == 'gold':
        return gold_dp_scorer(dataset_names)

    if dp_name == 'markov_n=3':

        from discourse_planning import make_database
        from discourse_planning import make_markov_scorer

        train_a, *resto = make_database('train')
        dev_a, *resto = make_database('dev')
        train_dev_a = list(flatten([train_a, dev_a]))

        return make_markov_scorer(train_dev_a, n=3)


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

    #return score_one_sentence_per_triple_best
    return scorer


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


def gold_sa_scorer(dataset_names):

    dev = load_dataset('dev')
    seen = set()
    i = 0

    def gold_sa_scorer_(sas, n_triples):

        nonlocal i
        nonlocal seen

        tripleset = tuple(sorted(flatten(sas[0])))

        aggs = []
        for l in dev[i].lexes:
            st = l['sorted_triples']
            if st:
                aggs.append([[x for x in y] for y in st])

        scores = []
        for sa in sas:
            if len(sa) == n_triples:
                scores.append(2)
            elif sa in aggs:
                scores.append(1)
            else:
                scores.append(0)

        if tripleset not in seen:
            i = i + 1
            seen.add(tripleset)

        return scores

    return gold_sa_scorer_


@lru_cache(maxsize=10)
def load_sentence_aggregation(dataset_names, sa_name):

    if sa_name == 'random':
        return random_sa_scorer
    if sa_name == 'ltr_lasso':
        return ltr_lasso_sa_scorer(dataset_names)
    if sa_name == 'inv_ltr_lasso':
        return inv_ltr_lasso_sa_scorer(dataset_names)
    # ! atualmente só funciona se o target for dev
    # !    e a geração dos textos for na agregação das entries no load_dev()
    if sa_name == 'gold':
        return gold_sa_scorer(dataset_names)


# Template Fallback
def load_template_fallback(dataset_names, fallback_name):

    if fallback_name == 'jjt':
        return JustJoinTemplate()


# Preprocessing
def load_preprocessing(preprocessing_name):

    if preprocessing_name == 'lower':
        return lambda t: t.lower()
