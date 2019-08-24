# -*- coding: utf-8 -*-
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter, defaultdict
from reading_thiagos_templates import make_template, get_lexicalizations
import pickle
import os
import spacy
from util import preprocess_so, Entry
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
from template_based2 import JustJoinTemplate
from itertools import islice
import re
from more_itertools import flatten
from teste3 import get_np


nlp = spacy.load('en_core_web_lg')


def preprocess_data(data, outdir='.'):

    # template

    dados = Counter()
    templates = []
    triple_to_lex_1 = defaultdict(list)
    triple_to_lex_gt1 = defaultdict(list)

    for entry in data:

        for lexe in entry.lexes:

            if lexe['comment'] == 'good' and entry.entity_map and lexe['template']:

                t = make_template(entry.triples,
                                  lexe['template'],
                                  entry.r_entity_map,
                                  metadata={})

                templates.append(t)

                pos_score0 = t.template_text.find('{slot0}')
                pos_score1 = t.template_text.find('{slot1}')

                is_active_voice = pos_score0 < pos_score1

                n_dots = max(1, t.template_text.count('.'))

                dados[(t, t.template_triples, entry.category, is_active_voice, n_dots)] += 1

                if len(entry.triples) > 1:
                    for triple in entry.triples:

                        triple_to_lex_gt1[triple].append(lexe['text'].lower())
                else:
                    for triple in entry.triples:

                        triple_to_lex_1[triple].append(lexe['text'].lower())


    template, templates_triples, categories, is_active_voices, ns_dots = zip(*dados.keys())
    df = pd.DataFrame({'feature_template_cnt_in_category': list(dados.values()),
                       'template': template,
                       'template_triples': templates_triples,
                       'feature_template_category': categories,
                       'feature_template_is_active_voice': is_active_voices,
                       'feature_template_n_dots': ns_dots})

    g = df.groupby(['template_triples', 'feature_template_category'])['feature_template_cnt_in_category'].sum()
    g.name = 'template_triples_and_category_cnt'
    g = g.reset_index()

    df = pd.merge(df, g)
    df['feature_template_freq_in_category'] = df['feature_template_cnt_in_category'] / df['template_triples_and_category_cnt']
    del df['template_triples_and_category_cnt']

    with open(os.path.join(outdir, 'triple_to_lex_1'), 'wb') as f:
        pickle.dump(triple_to_lex_1, f)

    with open(os.path.join(outdir, 'triple_to_lex_gt1'), 'wb') as f:
        pickle.dump(triple_to_lex_gt1, f)

    df.to_pickle(os.path.join(outdir, 'template_db'))
    df.to_csv(os.path.join(outdir, 'template_db.csv'), index=False)

    # reg

    name_db = defaultdict(lambda: Counter())
    pronoun_db = defaultdict(lambda: Counter())

    for entry in data:

        for lexe in entry.lexes:

            if lexe['comment'] == 'good' and entry.entity_map:

                lexicals = get_lexicalizations(lexe['text'],
                                               lexe['template'],
                                               entry.entity_map)

                if lexicals:

                    for lex_key, lex_values in lexicals.items():

                        for lex_value in lex_values:

                            doc = nlp(lex_value)

                            if len(doc) == 1 and doc[0].pos_ == 'PRON':

                                pronoun_db[lex_key][lex_value] += 1
                            else:
                                name_db[lex_key][lex_value] += 1

    name_db = dict(name_db)
    pronoun_db = dict(pronoun_db)

    with open(os.path.join(outdir, 'name_db'), 'wb') as f:
        pickle.dump(name_db, f)

    with open(os.path.join(outdir, 'pronoun_db'), 'wb') as f:
        pickle.dump(pronoun_db, f)


with open('../evaluation/train.pkl', 'rb') as f:
    train = pickle.load(f)


template_db = pd.read_pickle('./template_db')

np = NaiveDiscoursePlanFeature()
np.fit(template_db['template_triples'],
       template_db['feature_template_cnt_in_category'])
dp = DiscoursePlanning([np])
lpbpf = LessPartsBiggerFirst()
sa = SentenceAggregation([lpbpf])
ts = TemplateSelection(template_db, JustJoinTemplate())

npp = get_np(train)


with open('./name_db', 'rb') as f:
    name_db = pickle.load(f)

with open('./pronoun_db', 'rb') as f:
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

    plans = dp.plan(e)

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


TOKENIZER_RE = re.compile(r'(\W)')

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


def get_data(td, n1, n2, ranking):

    import pandas as pd

    data = everything(td, n1, n2, ranking)

    df = pd.DataFrame(data)

    del df['agg']
    del df['plan']
    del df['templates']

    return df


def cool_ranking(t):

    return (t['feature_template_n_fallback'] == 0,
            t['feature_agg_less_parts_bigger_first']*-1,
            t['feature_template_pct_same_category'],
            t['feature_template_len_1_freq'],
            t['feature_plan_naive_discourse_prob'],
            t['feature_template_template_freqs'])


# Training

#preprocess_data(train)

#df = get_data(train, 100, 10, cool_ranking)

#df.to_csv('./train_data.csv')
