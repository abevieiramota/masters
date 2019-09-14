# -*- coding: utf-8 -*-
from util import load_train_dev
import pickle
from collections import defaultdict
from more_itertools import partitions
import sentence_aggregation
import numpy as np


def calc_distance(agg, good_aggs):

    recalls = []

    for good_agg in good_aggs:

        recall = len(set(good_agg).intersection(agg)) / len(good_agg)

        recalls.append(recall)

    return max(recalls)


def make_data(entries):

    data_train = defaultdict(list)

    for e in entries:

        good_aggs = [l['sorted_triples'] for l in e.lexes
                     if l['comment'] == 'good' and l['sorted_triples'] and
                     sum(len(x)
                         for x in l['sorted_triples']) == len(e.triples)]

        if good_aggs:
            all_aggs = partitions(e.triples)

            for agg in all_aggs:

                agg = tuple([tuple(x) for x in agg])

                distance = calc_distance(agg, good_aggs)

                data = (agg, distance)

                data_train[len(e.triples)].append(data)

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

        ef = sentence_aggregation.SentenceAggregationFeatures().fit(X_raw, y)

        X = ef.transform(X_raw)

        data = np.c_[np.array(X), y]
        data = np.unique(data, axis=0)

        np.save(outpath + f'_{k}', data)

        extractors[k] = ef

    with open(outpath + '_extractors', 'wb') as f:
        pickle.dump(extractors, f)


def make_dataset():

    outpath = '../data/templates/sentence_aggregation_data'

    td = load_train_dev()

    make_main_model_data(td, outpath)
