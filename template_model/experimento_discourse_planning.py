# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from discourse_planning import get_scorer
import numpy as np
import pickle


def get_dp_scorer():

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

    scorer = get_scorer(models, fe)

    return scorer


def cool_ranking(t):

    return (t['feature_template_n_fallback'] == 0,
            t['feature_template_pct_same_category'],
            t['feature_agg_less_parts_bigger_first']*-1,
            t['feature_template_len_1_freq'],
            t['feature_template_n_max_precision'],
            t['feature_template_template_freqs'],
            t['feature_plan_naive_discourse_prob'],
            t['feature_template_template_freqs'])
