# -*- coding: utf-8 -*-
from util import load_train_dev, Entry
import pickle
from scipy.stats import kendalltau
from collections import defaultdict
from itertools import permutations
import discourse_planning
import numpy as np
from more_itertools import flatten


def calc_kendall(o1, good_os):

    all_kendall = [kendalltau(o, o1).correlation for o in good_os]

    max_kendall = max(all_kendall)

    return max_kendall


def extract_orders(e):

    orders = set()

    for l in [l for l in e.lexes if l['comment'] == 'good']:

        order = tuple(flatten(l['sorted_triples']))
        if len(order) == len(e.triples):
            orders.add(order)

    return list(orders)


def make_data(entries):

    data_train = defaultdict(list)

    for e in entries:

        good_orders = extract_orders(e)

        if good_orders:
            all_orders = permutations(e.triples)

            for o in all_orders:

                kendall = calc_kendall(o, good_orders)

                data = (o, kendall)

                data_train[len(o)].append(data)

    return data_train


def make_main_model_data(td, outpath):

    td_to_train_discourse_plan_ranker = [t
                                         for t in td
                                         if len(t.triples) > 1
                                         and t.r_entity_map]

    data = make_data(td_to_train_discourse_plan_ranker)

    extractors = {}

    for k, v in data.items():

        X_raw = [x[0] for x in v]
        y = [x[1] for x in v]

        ef = discourse_planning.ExtractFeatures().fit(X_raw, y)

        X = ef.transform(X_raw)

        data = np.c_[np.array(X), y]
        data = np.unique(data, axis=0)

        np.save(outpath + f'_{k}', data)

        extractors[k] = ef

    with open(outpath + '_extractors', 'wb') as f:
        pickle.dump(extractors, f)


def make_dataset():

    outpath = '../data/templates/discourse_plan_data'

    td = load_train_dev()

    make_main_model_data(td, outpath)
