# -*- coding: utf-8 -*-
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature, get_pipeline, get_sorter
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
from template_based2 import JustJoinTemplate
import pandas as pd
import pickle
from reg import REGer
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

with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
    name_db = pickle.load(f)

with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
    pronoun_db = pickle.load(f)


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
