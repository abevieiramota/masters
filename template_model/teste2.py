# -*- coding: utf-8 -*-
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
import pandas as pd
from template_based2 import JustJoinTemplate
from itertools import islice
import pickle
from util import preprocess_so


template_db = pd.read_pickle('../data/templates/template_db/template_db')

np = NaiveDiscoursePlanFeature()
np.fit(template_db['template_triples'],
       template_db['feature_template_cnt_in_category'])
dp = DiscoursePlanning(template_db, [np])
lpbpf = LessPartsBiggerFirst()
sa = SentenceAggregation([lpbpf])
ts = TemplateSelection(template_db, JustJoinTemplate())

with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
    name_db = pickle.load(f)

with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
    pronoun_db = pickle.load(f)


def lexicalize(s, ctx):

    if s in ctx['seen']:

        if s in pronoun_db:

            return pronoun_db[s].most_common()[0][0]
        else:
            return ''

    ctx['seen'].add(s)

    if s in name_db:

        return name_db[s].most_common()[0][0]
    else:
        return preprocess_so(s)


def get_templates(e):

    plans = list(dp.plan(e))

    plan_aggs = [(plan, sa.agg(plan['plan'])) for plan in plans]

    while plan_aggs:

        for plan, aggs in plan_aggs:

            try:
                agg = next(aggs)
            except StopIteration:
                plan_aggs.remove((plan, aggs))
                continue

            selects = ts.select(agg['agg'], e)

            for select in selects:

                result = {}
                result.update(plan)
                result.update(agg)
                result.update(select)

                yield result


def get_n_templates(e, n):

    return list(islice(get_templates(e), 0, n))


def get_best_template(e, n, ranking):

    return max(get_n_templates(e, n), key=ranking)


def get_n_best_template(e, n1, n2, ranking):

    return list(sorted(get_n_templates(e, n1),
                       key=ranking,
                       reverse=True))[:n2]


cool_ranking=lambda t: (t['feature_template_pct_same_category'],
                        t['feature_template_total_dots']*-1,
                        t['feature_template_template_freqs'] *
                        t['feature_plan_naive_discourse_prob'])


def get_best_text(e, n=1, ranking=lambda x: 1):

    tt = get_best_template(e, n, ranking)

    ctx = {'seen': set()}
    texts = [t.fill(a, lexicalize, ctx)
             for t, a in zip(tt['templates'], tt['agg'])]

    return ' '.join(texts)


with open('../evaluation/test.pkl', 'rb') as f:
    test = pickle.load(f)


def make_texts(n=1, ranking=lambda x: 1):

    i = 0

    with open('../data/models/abe-4/abe-4.txt', 'w', encoding='utf-8') as f:

        for e in test:

            t = get_best_text(e, n, ranking)

            f.write('{}\n'.format(t))

            i += 1

            if i == 100:
                f.flush()
                i = 0
