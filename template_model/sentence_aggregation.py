# -*- coding: utf-8 -*-
from more_itertools import partitions, flatten
from template_based import abstract_triples
from math import ceil
from collections import Counter
from functools import reduce


class LessPartsBiggerFirst:

    def extract_all(self, seqs):

        rank = sorted(range(len(seqs)),
                      key=lambda i: [len(seqs[i])] +
                                    [j*len(z)*-1
                                     for j, z in enumerate(seqs[i])])

        return [{'feature_agg_less_parts_bigger_first': i}
                for i in rank]


# don't generate an aggregation if there is any part w len > 1
#    that doesn't have a template
class SentenceAggregation:

    def __init__(self,
                 pct,
                 triples_to_templates,
                 feature_all_extractors=None,
                 sort=None):

        if not feature_all_extractors:
            feature_all_extractors = []

        self.triples_to_templates = triples_to_templates
        self.feature_all_extractors = feature_all_extractors
        self.sort = sort if sort else lambda x: list(x)
        self.pct = pct

    def agg(self, plan):

        all_partitions = self.sort(partitions(plan))
        all_partitions_w_templates = []

        n_max = ceil(self.pct * len(all_partitions))

        for partition in all_partitions:

            all_part_has_template = True
            for part in partition:

                if len(part) > 1:

                    a_part = abstract_triples(part)

                    if a_part not in self.triples_to_templates:
                        all_part_has_template = False
                        break

            if all_part_has_template:
                all_partitions_w_templates.append(partition)

        all_partitions_w_templates = all_partitions_w_templates[:n_max]

        aggs = [{'agg': agg} for agg in all_partitions_w_templates]

        for fae in self.feature_all_extractors:

            features = fae.extract_all(all_partitions)

            for a, f in zip(aggs, features):

                a.update(f)

        for agg in aggs:

            yield agg


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
