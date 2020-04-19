# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        Entry,
        abstract_triples
)
import pickle
from collections import defaultdict, Counter
from more_itertools import partitions, flatten
import numpy as np
import os
from functools import reduce


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')


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


def make_main_model_data(dataset_names):

    subsets_name = '_'.join(sorted(dataset_names))
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

        ef = SentenceAggregationFeatures().fit(X_raw, y)

        X = ef.transform(X_raw)

        data = np.c_[np.array(X), y]
        data = np.unique(data, axis=0)

        sa_data_filename = f'sa_data_{subsets_name}_{k}'
        sa_data_filepath = os.path.join(PRETRAINED_DIR, sa_data_filename)

        np.save(sa_data_filepath, data)

        extractors[k] = ef

    sa_extractor_filename = f'sa_extractor_{subsets_name}'
    sa_extractor_filepath = os.path.join(PRETRAINED_DIR, sa_extractor_filename)

    with open(sa_extractor_filepath, 'wb') as f:
        pickle.dump(extractors, f)


class SentenceAggregationFeatures:

    def fit(self, X, y=None):

        self.freq_parts = Counter()
        self.freq_partitions = Counter()

        for agg in X:

            a_agg = abstract_triples(flatten(agg))

            self.freq_partitions[a_agg] += 1

            for agg_part in agg:

                self.freq_parts[abstract_triples(agg_part)] += 1

        self.total_parts = sum(self.freq_parts.values())
        self.total_partitions = sum(self.freq_partitions.values())

        self.feature_names_ = ['pct_partition',
                               'pct_longest_partition',
                               'freq_parts',
                               'freq_partition']

        return self

    def transform(self, X, y=None):

        return [self.extract_features(o) for o in X]

    def extract_features(self, agg):

        a_agg = abstract_triples(flatten(agg))
        n_triples = sum(len(x) for x in agg)

        pct_partition = len(agg) / n_triples

        pct_longest_partition = max([len(x) for x in agg]) / n_triples

        freq_partition = self.freq_partitions[a_agg] / self.total_partitions

        freq_parts = reduce(lambda x, y: x*y,
                            [(self.freq_parts[abstract_triples(agg_part)] + 1) / (self.total_parts + n_triples)
                             for agg_part in agg])

        features = [pct_partition,
                    pct_longest_partition,
                    freq_parts,
                    freq_partition]

        return features
