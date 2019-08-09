# -*- coding: utf-8 -*-
from collections import Counter
from math import log
from reading_thiagos_templates import read_thiagos_xml_entries
import glob
from more_itertools import flatten


class NaiveProb:

    def __init__(self):

        self._counter = Counter()
        self._all_counts = 0
        self._vocab = set()

    def fit(self, seqs):

        for seq in seqs:

            padded_seq = ['<INI>'] + seq + ['<END>']

            for pair in zip(padded_seq[:-1], padded_seq[1:]):

                self._counter[pair] += 1
                self._counter[pair[0]] += 1
                self._counter[pair[1]] += 1
                self._vocab.add(pair[0])
                self._vocab.add(pair[1])

        self._vocab_len = len(self._vocab)

    def extract(self, seq):

        probs = []

        padded_seq = ['<INI>'] + seq + ['<END>']

        for pair in zip(padded_seq[:-1], padded_seq[1:]):

            prob_pair = (self._counter[pair] + 1) / \
                        (self._counter[pair[0]] +
                         self._vocab_len)

            probs.append(log(prob_pair))

        return sum(probs)


def get_np():

    food_files = glob.glob('../data/templates/v1.4/train/**/Food.xml')

    for food_file in food_files:

        es = read_thiagos_xml_entries(food_file)

        texts = [[l['text'] for l in e['lexes']] for e in es]

        texts = list(flatten(texts))

    np = NaiveProb()
    np.fit([t.split() for t in texts])

    return np
