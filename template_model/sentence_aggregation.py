# -*- coding: utf-8 -*-
from more_itertools import partitions
from template_based2 import abstract_triples


class LessPartsBiggerFirst:

    def extract_all(self, seqs):

        rank = sorted(range(len(seqs)),
                      key=lambda i: [len(seqs[i])] +
                                    [j*len(z)*-1
                                     for j, z in enumerate(seqs[i])])

        return [{'feature_agg_less_parts_bigger_first': i}
                for i in rank]


# don't generate an aggregation if there is any part w len > 1 that doesn't have a template
class SentenceAggregation:

    def __init__(self, triples_to_templates, feature_all_extractors=None, sort=None):

        if not feature_all_extractors:
            feature_all_extractors = []

        self.triples_to_templates = triples_to_templates
        self.feature_all_extractors = feature_all_extractors
        self.sort = sort if sort else lambda x: list(x)

    def agg(self, plans):

        all_partitions = self.sort(partitions(plans))
        all_partitions_w_templates = []

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

        aggs = [{'agg': agg} for agg in all_partitions_w_templates]

        for fae in self.feature_all_extractors:

            features = fae.extract_all(all_partitions)

            for a, f in zip(aggs, features):

                a.update(f)

        for agg in aggs:

            yield agg
