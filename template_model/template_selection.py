# -*- coding: utf-8 -*-
from template_based2 import abstract_triples
from collections import defaultdict
from itertools import product
from operator import mul
from functools import reduce


class TemplateSelection:

    def __init__(self, template_db, fallback=None):

        self.fallback = fallback
        triples_to_templates = defaultdict(list)

        for v in template_db.to_dict(orient='record'):

            triples_to_templates[v['template_triples']].append(v)

        self.triples_to_templates = dict(triples_to_templates)

    def select(self, agg, e):

        templates = []

        for agg_part in agg:

            abstracted_triples = abstract_triples(agg_part)

            if abstracted_triples in self.triples_to_templates:

                ts = self.triples_to_templates[abstracted_triples]
                ts = sorted(ts,
                            key=lambda t:
                            (t['feature_template_category'] == e.category,
                             t['feature_template_is_active_voice']),
                            reverse=True)

                templates.append([(agg_part, t, False)
                                  for t in ts])
            else:

                for triple in agg_part:

                    templates.append(([([triple],
                                        {'template': self.fallback,
                                         'feature_template_freq_in_category': 0,
                                         'feature_template_category': None,
                                         'feature_template_n_dots': 1
                                         },
                                        True)]))

        for item in product(*templates):

            result = {}
            result['agg'] = [i[0] for i in item]
            result['templates'] = [i[1]['template'] for i in item]
            result['feature_template_n_fallback'] = sum((i[2] for i in item))
            result['feature_template_template_freqs'] = reduce(
                    mul,
                    (i[1]['feature_template_freq_in_category'] for i in item))
            result['feature_template_pct_same_category'] = \
                sum(i[1]['feature_template_category'] == e.category
                    for i in item) / len(item)

            result['feature_template_total_dots'] = \
                sum(i[1]['feature_template_n_dots'] for i in item)

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
