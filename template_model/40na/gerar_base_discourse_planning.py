# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        Entry,
        abstract_triples
)
from util import extract_orders
import pickle
from scipy.stats import kendalltau
from collections import defaultdict, Counter
from itertools import permutations
import numpy as np
import os
from functools import reduce
from more_itertools import flatten


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')


def calc_kendall(o1, good_os):

    all_kendall = [kendalltau(o, o1).correlation for o in good_os]

    max_kendall = max(all_kendall)

    return max_kendall


def make_data(entries):

    data_train = defaultdict(list)

    for e in entries:

        good_orders = [o for o in extract_orders(e) if o]

        if good_orders:
            all_orders = permutations(e.triples)

            for o in all_orders:

                kendall = calc_kendall(o, good_orders)

                data = (o, kendall)

                data_train[len(o)].append(data)

    return data_train


def make_main_model_data(dataset_names):

    subset_names = '_'.join(sorted(dataset_names))

    data = list(flatten(load_dataset(d) for d in dataset_names))

    data_to_train_discourse_plan_ranker = [t
                                           for t in data
                                           if len(t.triples) > 1
                                           and t.entity_map]

    data = make_data(data_to_train_discourse_plan_ranker)

    extractors = {}

    for k, v in data.items():

        X_raw = [x[0] for x in v]
        y = [x[1] for x in v]

        ef = DiscoursePlanningFeatures().fit(X_raw, y)

        X = ef.transform(X_raw)

        data = np.c_[np.array(X), y]
        data = np.unique(data, axis=0)

        dp_data_filename = f'dp_data_{subset_names}_{k}'
        dp_data_filepath = os.path.join(PRETRAINED_DIR, dp_data_filename)

        np.save(dp_data_filepath, data)

        extractors[k] = ef

    dp_extractor_filename = f'dp_extractor_{subset_names}'
    dp_extractor_filepath = os.path.join(PRETRAINED_DIR, dp_extractor_filename)

    with open(dp_extractor_filepath, 'wb') as f:
        pickle.dump(extractors, f)


def frac_siblings_subjects(o):

    i = 0
    current_s = None
    for o_ in o:
        if o_.subject == current_s:
            i += 1
        else:
            current_s = o_.subject

    subs = Counter(o_.subject for o_ in o)
    max_siblings = sum(v - 1 for v in subs.values())

    return (i + 1) / (max_siblings + len(o))


def frac_siblings_objects(o):

    i = 0
    current_o = None
    for o_ in o:
        if o_.object == current_o:
            i += 1
        else:
            current_o = o_.object

    obs = Counter(o_.object for o_ in o)
    max_siblings = sum(v - 1 for v in obs.values())

    return (i + 1) / (max_siblings + len(o))


def frac_siblings_predicates(o):

    i = 0
    current_p = None
    for o_ in o:
        if o_.predicate == current_p:
            i += 1
        else:
            current_p = o_.predicate

    preds = Counter(o_.predicate for o_ in o)
    max_siblings = sum(v - 1 for v in preds.values())

    return (i + 1) / (max_siblings + len(o))


def frac_chains(o):

    n = 0
    for t, t_1 in zip(o[:-1], o[1:]):
        if t.object == t_1.subject:
            n += 1

    subs = set(t.subject for t in o)
    objs = set(t.object for t in o)

    len_intersect = len(subs.intersection(objs))

    return (n + 1) / (len_intersect + len(o))


def is_first_the_main(o):

    subjs = set(t.subject for t in o)
    objs = set(t.object for t in o)

    s_not_o = subjs - objs

    return o[0].subject in s_not_o


class DiscoursePlanningFeatures:

    def fit(self, X, y=None):

        self.freq_all = Counter()
        self.freq_bigrams = Counter()
        self.freq_unigrams = Counter()
        self.size_training = Counter()

        good_orders = [x for i, x in enumerate(X) if y[i] == 1.0]

        for o in good_orders:
            for pos, t in enumerate(o):
                self.freq_unigrams[t.predicate] += 1

            a_o = abstract_triples(o)

            self.freq_all[a_o] += 1

            for t1, t2 in zip(o[:-1], o[1:]):
                self.freq_bigrams[(t1.predicate, t2.predicate)] += 1

            self.size_training[len(o)] += 1

        self.feature_names_ = ['freq_all',
                               'frac_chains',
                               'is_first_the_main',
                               'frac_siblings_subjects',
                               'frac_siblings_objects',
                               'frac_siblings_predicates',
                               'naive_prob_predicates']

        return self

    def transform(self, X, y=None):

        return [self.extract_features(o) for o in X]

    def extract_features(self, order):

        a_o = abstract_triples(order)

        naive_prob_predicates = reduce(lambda x, y: x*y,
                                       [(self.freq_bigrams[(t1.predicate, t2.predicate)] + 1) / (self.freq_unigrams[t1.predicate] + len(order))
                                        for t1, t2 in zip(order[:-1], order[1:])])

        features = [self.freq_all[a_o] / self.size_training[len(order)],
                    frac_chains(order),
                    is_first_the_main(order),
                    frac_siblings_subjects(order),
                    frac_siblings_objects(order),
                    frac_siblings_predicates(order),
                    naive_prob_predicates]

        return features
