# -*- coding: utf-8 -*-
from discourse_planning import (
        NaiveDiscourseFeature,
        SameOrderDiscoursePlanning)
from sentence_aggregation import PartitionsSentenceAggregation
from template_selection import (
        MostFrequentTemplateSelection,
        AllTemplateSelection
        )
from template_based2 import TemplateBasedModel, JustJoinTemplate
from util import preprocess_so
import pandas as pd
import pickle


with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
    name_db = pickle.load(f)

with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
    pronoun_db = pickle.load(f)

template_db = pd.read_pickle('../data/templates/template_db/template_db')


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


def get_full_model(n):

    dp = NaiveDiscoursePlanning()
    dp.fit(template_db['template_triples'],
           template_db['cnt'])

    sa = PartitionsSentenceAggregation()

    ts = MostFrequentTemplateSelection(n, template_db, JustJoinTemplate())

    tbm = TemplateBasedModel(dp, sa, ts, lexicalize)

    return tbm


def get_basic_model(n):

    dp = SameOrderDiscoursePlanning()

    sa = PartitionsSentenceAggregation()

    ts = MostFrequentTemplateSelection(n, template_db, JustJoinTemplate())

    tbm = TemplateBasedModel(dp, sa, ts, lexicalize)

    return tbm


def get_over_model():

    dp = SameOrderDiscoursePlanning()

    sa = PartitionsSentenceAggregation()

    ts = AllTemplateSelection(template_db, JustJoinTemplate())

    tbm = TemplateBasedModel(dp, sa, ts, lexicalize)

    return tbm


def get_cool_ranking():

    return lambda texts: sorted(texts, key=lambda t: (t['n_fallback'],
                                                      len(t['agg'])))
