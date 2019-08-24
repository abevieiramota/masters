# -*- coding: utf-8 -*-
from collections import Counter
from math import log
import re

TOKENIZER_RE = re.compile(r'(\W)')


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

    def extract(self, t):

        if t is None:
            return 0

        seq = TOKENIZER_RE.split(t.lower())

        probs = []

        padded_seq = ['<INI>'] + seq + ['<END>']

        for pair in zip(padded_seq[:-1], padded_seq[1:]):

            prob_pair = (self._counter[pair] + 1) / \
                        (self._counter[pair[0]] +
                         self._vocab_len)

            probs.append(log(prob_pair))

        return sum(probs)


def get_np(td):

    all_texts = []

    for e in td:

        texts = [l['text'] for l in e.lexes if l['text']]

        all_texts.extend(texts)

    np = NaiveProb()
    np.fit([TOKENIZER_RE.split(t.lower()) for t in all_texts])

    return np
