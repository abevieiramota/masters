# -*- coding: utf-8 -*-
from discourse_planning import (
        DiscoursePlanning,
        NaiveDiscoursePlanFeature,
        get_sorter
        )
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
from template_based import JustJoinTemplate
import pandas as pd
import pickle
from reg import REGer, load_name_db, load_pronoun_db
from collections import defaultdict
from random import shuffle
from model import TextGenerationModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import numpy as np


template_db = pd.read_pickle('../data/templates/template_db/template_db')

triples_to_templates = defaultdict(list)

for v in template_db.to_dict(orient='record'):

    triples_to_templates[v['template_triples']].append(v)

triples_to_templates = dict(triples_to_templates)

ndp = NaiveDiscoursePlanFeature()
ndp.fit(template_db['template_triples'],
        template_db['feature_template_cnt_in_category'])

pronoun_db = load_pronoun_db()
name_db = load_name_db()


def get_model(pct, sorter):

    dp = DiscoursePlanning(pct, [ndp], sort=sorter)

    # Sentence Aggregation
    lpbpf = LessPartsBiggerFirst()
    sa = SentenceAggregation(triples_to_templates, [lpbpf])

    # Template Selection
    ts = TemplateSelection(triples_to_templates, JustJoinTemplate())

    # REG
    refer = REGer(pronoun_db, name_db).refer

    model = TextGenerationModel(dp, sa, ts, refer)

    return model


def get_random_model(pct):
    # Discourse Planning

    def random_order(orders, e):

        orders_ = list(orders)[::]
        shuffle(orders_)

        return orders_

    return get_model(pct, random_order)


def get_main_model(pct):

    models = {}

    with open('../data/templates/discourse_plan_data_extractors', 'rb') as f:
        fe = pickle.load(f)

    for i in range(2, 8):

        data = np.load(f'../data/templates/discourse_plan_data_{i}.npy')

        X = data[:, :-1]
        y = data[:, -1]

        pipe = Pipeline([
            ('mms',  MinMaxScaler()),
            ('clf', Lasso(alpha=0.0001))
        ])

        pipe.fit(X, y)

        models[i] = pipe

    sorter = get_sorter(models, fe)

    return get_model(pct, sorter)


def cool_ranking(t):

    return (t['feature_template_n_fallback'] == 0,
            t['feature_template_pct_same_category'],
            t['feature_agg_less_parts_bigger_first']*-1,
            t['feature_template_len_1_freq'],
            t['feature_template_n_max_precision'],
            t['feature_template_template_freqs'],
            t['feature_plan_naive_discourse_prob'],
            t['feature_template_template_freqs'])
