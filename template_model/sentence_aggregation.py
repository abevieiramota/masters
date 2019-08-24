# -*- coding: utf-8 -*-
from more_itertools import partitions


class LessPartsBiggerFirst:

    def extract_all(self, seqs):

        rank = sorted(range(len(seqs)),
                      key=lambda i: [len(seqs[i])] +
                                    [j*len(z)*-1
                                     for j, z in enumerate(seqs[i])])

        return [{'feature_agg_less_parts_bigger_first': i}
                for i in rank]


class SentenceAggregation:

    def __init__(self, feature_all_extractors=None, sort=None):

        if not feature_all_extractors:
            feature_all_extractors = []

        self.feature_all_extractors = feature_all_extractors
        self.sort = sort if sort else lambda x: list(x)

    def agg(self, plans):

        all_partitions = self.sort(partitions(plans))

        aggs = [{'agg': agg} for agg in all_partitions]

        for fae in self.feature_all_extractors:

            features = fae.extract_all(all_partitions)

            for a, f in zip(aggs, features):

                a.update(f)

        for agg in aggs:

            yield agg
