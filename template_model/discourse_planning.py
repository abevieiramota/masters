# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from itertools import permutations


class SameOrderDiscoursePlanning:

    def plan(self, e):

        return [e.triples]


class NaiveDiscoursePlanning:

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

    def prob(self, triples):

        prob = 1
        seq = [t.predicate for t in triples]
        seq_len = len(seq)

        for pair in zip(seq[:-1], seq[1:]):

            prob_pair = (self._counters[seq_len][pair] + 1) / \
                        (self._counters[seq_len][pair[0]] +
                         self._vocabs_lens[seq_len])

            prob *= prob_pair

        return prob

    def plan(self, e):

        perm_seqs = [(perm, self.prob(perm))
                     for perm in permutations(e.triples)]

        for perm, prob in sorted(perm_seqs, key=lambda v: v[1], reverse=True):

            yield perm
