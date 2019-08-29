# -*- coding: utf-8 -*-
from itertools import islice
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
from template_based2 import JustJoinTemplate
import pandas as pd
import pickle
from reg import REGer


class TextGenerationModel:

    def __init__(self, dp, sa, ts, refer):
        self.dp = dp
        self.sa = sa
        self.ts = ts
        self.refer = refer

    def get_templates(self, e):

        plans = self.dp.plan(e)

        plan_aggs = [(plan, self.sa.agg(plan['plan'])) for plan in plans]

        while plan_aggs:

            for plan, aggs in plan_aggs:

                try:
                    agg = next(aggs)
                except StopIteration:
                    plan_aggs.remove((plan, aggs))
                    continue

                selects = self.ts.select(agg['agg'], e)

                for select in selects:

                    result = {}
                    result.update(plan)
                    result.update(agg)
                    result.update(select)

                    yield result

    def get_n_templates(self, e, n):

        return list(islice(self.get_templates(e), 0, n))

    def get_best_template(self, e, n, ranking):

        return max(self.get_n_templates(e, n), key=ranking)

    def get_n_best_template(self, e, n1, n2, ranking):

        return list(sorted(self.get_n_templates(e, n1),
                           key=ranking,
                           reverse=True))[:n2]

    def get_n_best_texts(self, e, n1, n2, ranking):

        templates = self.get_n_best_template(e, n1, n2, ranking)

        for tt in templates:

            ctx = {'seen': set()}
            texts = [t.fill(a, self.refer, ctx)
                     for t, a in zip(tt['templates'], tt['agg'])]

            tt['text'] = ' '.join(texts)

        return templates

    def get_best_text(self, e, n=1, ranking=lambda x: 1):

        tt = self.get_best_template(e, n, ranking)

        ctx = {'seen': set()}
        texts = [t.fill(a, self.refer, ctx)
                 for t, a in zip(tt['templates'], tt['agg'])]

        return ' '.join(texts)

    def make_texts(self, X, n=1, ranking=lambda x: 1, outfilepath=None):

        with open(outfilepath, 'w', encoding='utf-8') as f:

            for e in X:

                t = self.get_best_text(e, n, ranking)

                f.write('{}\n'.format(t))


def get_model():

    # Discourse Planning
    template_db = pd.read_pickle('../data/templates/template_db/template_db')

    np = NaiveDiscoursePlanFeature()
    np.fit(template_db['template_triples'],
           template_db['feature_template_cnt_in_category'])
    dp = DiscoursePlanning([np])

    # Sentence Aggregation
    lpbpf = LessPartsBiggerFirst()
    sa = SentenceAggregation([lpbpf])

    # Template Selection
    ts = TemplateSelection(template_db, JustJoinTemplate())

    # REG
    with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
        name_db = pickle.load(f)

    with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
        pronoun_db = pickle.load(f)

    refer = REGer(pronoun_db, name_db).refer

    model = TextGenerationModel(dp, sa, ts, refer)

    return model


def cool_ranking(t):

    return (t['feature_template_n_fallback'] == 0,
            t['feature_template_pct_same_category'],
            t['feature_agg_less_parts_bigger_first']*-1,
            t['feature_template_len_1_freq'],
            t['feature_template_n_max_precision'],
            t['feature_template_template_freqs'],
            t['feature_plan_naive_discourse_prob'],
            t['feature_template_template_freqs'])