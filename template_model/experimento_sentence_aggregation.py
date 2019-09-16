# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from discourse_planning import get_sorter
from more_itertools import flatten


def get_sa_sorter():

    models = {}

    with open('../data/templates/sentence_aggregation_data_extractors', 'rb') as f:
        fe = pickle.load(f)

    for i in range(2, 8):

        data = np.load(f'../data/templates/sentence_aggregation_data_{i}.npy')

        X = data[:, :-1]
        y = data[:, -1]

        pipe = Pipeline([
            ('mms',  MinMaxScaler()),
            ('clf', Lasso(alpha=0.0001))
        ])

        pipe.fit(X, y)

        models[i] = pipe

    sorter = get_sorter(models, fe)

    def sort_one_sentence_per_triple_first(os, flow_chain):

        os_rest = [o for o in os if any(len(agg_part) > 1 for agg_part in os)]
        os_ospt = [[t] for t in flatten(os[0])]

        sorted_rest = sorter(os_rest, flow_chain)

        return [os_ospt] + sorted_rest

    return sort_one_sentence_per_triple_first
