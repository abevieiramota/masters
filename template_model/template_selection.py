# -*- coding: utf-8 -*-
from template_based2 import abstract_triples
from collections import defaultdict
from itertools import product
from operator import mul
from functools import reduce
import pickle
from nltk import everygrams
import numpy as np


with open('../data/templates/template_db/triple_to_lex_1', 'rb') as f:
    triple_to_lex_1 = pickle.load(f)

with open('../data/templates/template_db/triple_to_lex_gt1', 'rb') as f:
    triple_to_lex_gt1 = pickle.load(f)


def super_sim(agg_part, t):

    t_grams = set(everygrams(t.split(), 2, 4))

    if len(agg_part) == 1:

        lexes = triple_to_lex_gt1[agg_part[0]]

        if len(lexes) == 0:
            return (0, 1), 0

        lexes_ngrams = [set(everygrams(lexe.split(), 2, 4)) for lexe in lexes]

        intersections = [len(t_grams.intersection(lexe_ngrams))
                         for lexe_ngrams in lexes_ngrams]

        max_intersection = max(intersections)
        n_max_intersections = len([p for p in intersections if p == max_intersection])

        return (max_intersection, len(t_grams)), n_max_intersections
    else:

        each_intersections = []
        n_max_precisions = []
        each_sizes = []

        for a in agg_part:

            lexes = triple_to_lex_1[a]

            if len(lexes) == 0:
                each_intersections.append(0)
                each_sizes.append(0)
                continue

            lexes_ngrams = [set(everygrams(lexe.split(), 2, 4)) for lexe in lexes]

            intersections = [len(t_grams.intersection(lexe_ngrams))
                             for lexe_ngrams in lexes_ngrams]

            sizes = [len(lexe_ngrams) for lexe_ngrams in lexes_ngrams]

            precisions = [i/s for i, s in zip(intersections, sizes)]

            i_max_precision = max(range(len(intersections)),
                                  key=lambda i: intersections[i]/sizes[i])

            max_precision = precisions[i_max_precision]
            n_max_precision = len([p for p in precisions if p == max_precision])

            each_intersections.append(intersections[i_max_precision])
            each_sizes.append(intersections[i_max_precision])

            n_max_precisions.append(n_max_precision)

        sum_intersections = sum(each_intersections)
        sum_sizes = sum(each_sizes)

        if sum_sizes == 0:
            return (0, 1), 0

        return (sum_intersections, sum_sizes), np.median(n_max_precisions)


class TemplateSelection:

    def __init__(self, triples_to_templates, fallback=None, ranker=None):

        self.fallback = fallback
        self.triples_to_templates = triples_to_templates
        self.ranker = ranker if ranker else lambda x: list(x)

    def select(self, agg, e):

        templates = []

        for agg_part in agg:

            abstracted_triples = abstract_triples(agg_part)

            if abstracted_triples in self.triples_to_templates:

                ts = self.triples_to_templates[abstracted_triples]
                ts = self.ranker(ts)

                tems = []

                for t in ts:

                    tt = dict(t)
                    (intersection, size), n_max = super_sim(agg_part,
                                                            t['template'].template_text)
                    tt['feature_template_intersection'] = intersection
                    tt['feature_template_size'] = size
                    tt['feature_template_n_max_precision'] = n_max

                    tems.append((agg_part, tt, False))


                templates.append(tems)
            else:

                if len(abstracted_triples) == 1:

                    triple = agg_part[0]

                    data = {'template': self.fallback,
                            'feature_template_freq_in_category': 0,
                            'feature_template_category': None,
                            'feature_template_n_dots': 1,
                            'feature_template_len_1_freq': 0,
                            'feature_template_n_max_precision': 0,
                            'feature_template_intersection': 0,
                            'feature_template_size': 1
                            }

                    templates.append([([triple], data, True)])
                else: # no template for the agg w len > 1
                    return


        for item in product(*templates):

            result = {}
            result['agg'] = [i[0] for i in item]
            result['templates'] = [i[1]['template'] for i in item]
            result['feature_template_n_fallback'] = sum((i[2] for i in item)) / len(e.triples)
            result['feature_template_template_freqs'] = reduce(
                    mul,
                    (i[1]['feature_template_freq_in_category'] for i in item))
            result['feature_template_pct_same_category'] = \
                sum(i[1]['feature_template_category'] == e.category
                    for i in item) / len(item)

            result['feature_template_total_dots'] = \
                sum(i[1]['feature_template_n_dots'] for i in item)

            result['feature_template_len_1_freq'] = sum(i[1]['feature_template_intersection'] for i in item) / sum(i[1]['feature_template_size'] for i in item)
            result['feature_template_n_max_precision'] = sum(i[1]['feature_template_n_max_precision'] for i in item) / len(item)

            yield result


class MostFrequentTemplateSelection:

    def __init__(self, n, template_db, fallback):

        self.n = n
        self.fallback = fallback
        # TODO: stop using pandas...
        data_triples_cat_templates = template_db\
            .sort_values('cnt', ascending=False)\
            .drop_duplicates(['template_triples', 'category'])\
            .to_dict(orient='record')

        self.triples_cat_templates = {(t['template'].template_triples,
                                       t['category']): t['template']
                                      for t in data_triples_cat_templates}

        data_triples_templates = template_db\
            .sort_values('cnt', ascending=False)\
            .drop_duplicates(['template_triples'])\
            .to_dict(orient='record')

        self.triples_templates = {t['template'].template_triples: t['template']
                                  for t in data_triples_templates}

    def select(self, e, triples_agg):

        result = []
        n_fallback = 0

        for agg in triples_agg:

            abstracted_triples = abstract_triples(agg)

            if (abstracted_triples,
                    e.category) in self.triples_cat_templates:

                template = self.triples_cat_templates[(abstracted_triples,
                                                       e.category)]
                result.append((agg, template))
            elif abstracted_triples in self.triples_templates:

                template = self.triples_templates[abstracted_triples]
                result.append((agg, template))
            else:
                n_fallback += len(agg)
                for triple in agg:

                    result.append(([triple], self.fallback))

        return result, n_fallback
