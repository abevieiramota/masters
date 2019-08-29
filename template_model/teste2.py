# -*- coding: utf-8 -*-
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature, SameOrderDiscoursePlanning
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
import pandas as pd
from template_based2 import JustJoinTemplate
from itertools import islice
import pickle
from util import Entry
import re
from more_itertools import flatten
from teste3 import get_np
from reg import REGer




with open('../evaluation/train_dev.pkl', 'rb') as f:
    td = pickle.load(f)

npp = get_np(td)


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
        hyp['feature_language_model_score'] = npp.extract(hyp['text'])

    first_hyp = hypothesis[0]
    best_bleu = max(hypothesis, key=lambda h: h['bleu'])['bleu']

    first_hyp['feature_rank_is_first_ranked'] = True
    first_hyp['feature_rank_is_first_correct'] = first_hyp['bleu'] == best_bleu

    return hypothesis


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


def get_data(td, n1, n2, ranking):

    import pandas as pd

    data = everything(td, n1, n2, ranking)

    df = pd.DataFrame(data)

    del df['agg']
    del df['plan']
    del df['templates']

    return df



"""
plan_aggs = [(plan, sa.agg(plan['plan'])) for plan in plans]

for _, ags in plan_aggs:
    for ag in ags:
        tss = []
        for ag_part in ag['agg']:
            a_t = abstract_triples(ag_part)
            if a_t in ts.triples_to_templates:
                tss.append(len(ts.triples_to_templates[a_t]))
            else:
                tss.append(1)
        from functools import reduce
        total = reduce(lambda x, y: x*y, tss)
        i += total
        ttss.append(tss)
"""

def get_model_scores(e, n1, n2, ranking, reg):

    texts = get_n_best_texts(e, n1, n2, ranking)

    x = pd.DataFrame(texts)

    del x['plan']
    del x['agg']
    del x['text']
    del x['templates']

    ys = reg.predict(x)

    for tt, y in zip(texts, ys):
        tt['predicted'] = y

    return texts
