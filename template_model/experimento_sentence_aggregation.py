# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from discourse_planning import get_scorer
from more_itertools import flatten


def get_sa_scorer():

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

    scorer = get_scorer(models, fe)

    def score_one_sentence_per_triple_best(os, flow_chain):

        ix_one_sen_per_triple = [i for i in range(len(os)) if len(os[i]) == len(flow_chain[0].triples)][0]
        scores = scorer(os, flow_chain)

        scores[ix_one_sen_per_triple] = max(scores) + 1

        return scores

    return score_one_sentence_per_triple_best
