# -*- coding: utf-8 -*-
from template_based2 import abstract_triples


class NthFrequentTemplateSelection:

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
