# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from itertools import permutations


class NaiveDiscoursePlanFeature:

    def __init__(self):

        self._counters = defaultdict(lambda: Counter())
        self._all_counts = defaultdict(int)
        self._vocabs = defaultdict(set)

    def fit(self, triplesets, counts):

        for tripleset, count in zip(triplesets, counts):

            seq_len = len(tripleset)

            seq = [t.predicate for t in tripleset]

            padded_seq = ['<INI>'] + seq + ['<END>']

            for pair in zip(padded_seq[:-1], padded_seq[1:]):

                self._counters[seq_len][pair] += count
                self._counters[seq_len][pair[0]] += count
                self._counters[seq_len][pair[1]] += count
                self._vocabs[seq_len].add(pair[0])
                self._vocabs[seq_len].add(pair[1])

    def extract(self, triples):

        prob = 1
        seq = [t.predicate for t in triples]
        seq_len = len(seq)

        padded_seq = ['<INI>'] + seq + ['<END>']

        for pair in zip(padded_seq[:-1], padded_seq[1:]):

            prob_pair = (self._counters[seq_len][pair] + 1) / \
                        (self._counters[seq_len][pair[0]] +
                         len(triples))

            prob *= prob_pair

        return {'feature_plan_naive_discourse_prob': prob}

    def sort(self, l_triples):

        return sorted(l_triples,
                      key=lambda t:
                      self.extract(t)['feature_plan_naive_discourse_prob'])


class DiscoursePlanning:

    def __init__(self, feature_extractors=None, sort=None):

        if not feature_extractors:
            feature_extractors = []

        self.feature_extractors = feature_extractors
        self.sort = sort if sort else lambda x: list(x)

    def plan(self, e):

        plans = self.sort(permutations(e.triples))

        for i, plan in enumerate(plans):

            plan_f = {'plan': plan,
                      'feature_plan_is_first': i == 0}

            for fe in self.feature_extractors:

                plan_f.update(fe.extract(plan))

            yield plan_f
