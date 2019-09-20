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
