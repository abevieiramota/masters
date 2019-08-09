# -*- coding: utf-8 -*-
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
import pandas as pd
from template_based2 import JustJoinTemplate
from itertools import islice
import pickle
from util import preprocess_so, Entry
import re
from more_itertools import flatten
from teste3 import get_np


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

with open('../evaluation/train_dev.pkl', 'rb') as f:
    td = pickle.load(f)

np = get_np(td)


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


def get_n_best_texts(e, n1, n2, ranking):

    templates = get_n_best_template(e, n1, n2, ranking)

    for tt in templates:

        ctx = {'seen': set()}
        texts = [t.fill(a, lexicalize, ctx)
                 for t, a in zip(tt['templates'], tt['agg'])]

        tt['text'] = ' '.join(texts)

    return templates


def everything(td, n1, n2, ranking):

    result = []

    for e in td:

        e_result = get_sentence_bleu_scored(e, n1, n2, ranking)

        result.append(e_result)

    return list(flatten(result))


TOKENIZER_RE = re.compile(r'(\W)')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normtokenize(t):

    # someone is returning None
    # TODO: who?!
    if t is None:
        return []

    return ' '.join(TOKENIZER_RE.split(t.lower())).split()


def get_sentence_bleu_scored(e, n1, n2, ranking):

    hypothesis = get_n_best_texts(e, n1, n2, ranking)

    references = [l['text'] for l in e.lexes]

    ref_tokens = [normtokenize(ref) for ref in references]

    for hyp in hypothesis:

        hyp_tokens = normtokenize(hyp['text'])

        hyp['bleu'] = sentence_bleu(ref_tokens,
                                    hyp_tokens,
                                    smoothing_function=SmoothingFunction().method2)

        hyp['feature_entry_n_triples'] = len(e.triples)
        hyp['feature_entry_category'] = e.category
        hyp['feature_language_model_score'] = np.extract(hyp['text'])

    first_hyp = hypothesis[0]
    best_bleu = max(hypothesis, key=lambda h: h['bleu'])['bleu']

    first_hyp['feature_rank_is_first_ranked'] = True
    first_hyp['feature_rank_is_first_correct'] = first_hyp['bleu'] == best_bleu

    return hypothesis


def get_data(td, n1, n2, ranking):

    import pandas as pd

    data = everything(td, n1, n2, ranking)

    df = pd.DataFrame(data)

    del df['agg']
    del df['plan']
    del df['templates']

    return df


def cool_ranking(t):

    if len(t['agg']) == 1 and t['feature_template_n_fallback'] == 0:
        if t['feature_plan_is_first']:
            first_value = 10000*(1/t['feature_template_total_dots'])
        else:
            first_value = 1000*(1/t['feature_template_total_dots'])
    else:
        first_value = 0

    return (first_value,
            t['feature_template_pct_same_category'],
            (t['feature_template_template_freqs']) *
            (t['feature_plan_naive_discourse_prob']))



def get_best_text(e, n=1, ranking=lambda x: 1):

    tt = get_best_template(e, n, ranking)

    ctx = {'seen': set()}
    texts = [t.fill(a, lexicalize, ctx)
             for t, a in zip(tt['templates'], tt['agg'])]

    return ' '.join(texts)


with open('../evaluation/test.pkl', 'rb') as f:
    test = pickle.load(f)


def make_texts(n=1, ranking=lambda x: 1):

    with open('../data/models/abe/abe.txt', 'w', encoding='utf-8') as f:

        for e in test:

            t = get_best_text(e, n, ranking)

            f.write('{}\n'.format(t))
