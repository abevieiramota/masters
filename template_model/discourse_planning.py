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

        self._vocabs_lens = {seq_len: len(v)
                             for seq_len, v in self._vocabs.items()}

    def extract(self, triples):

        prob = 1
        seq = [t.predicate for t in triples]
        seq_len = len(seq)

        padded_seq = ['<INI>'] + seq + ['<END>']

        for pair in zip(padded_seq[:-1], padded_seq[1:]):

            prob_pair = (self._counters[seq_len][pair] + 1) / \
                        (self._counters[seq_len][pair[0]] +
                         self._vocabs_lens[seq_len])

            prob *= prob_pair

        return {'feature_plan_naive_discourse_prob': prob}


class DiscoursePlanning:

    def __init__(self, template_db, feature_extractors):

        self.feature_extractors = feature_extractors

    def plan(self, e):

        for i, perm in enumerate(permutations(e.triples)):

            plan_f = {'plan': perm,
                      'feature_plan_is_first': i == 0}

            for fe in self.feature_extractors:

                plan_f.update(fe.extract(perm))

            yield plan_f
