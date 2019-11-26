# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset
from more_itertools import flatten


def extract_orders(e):

    orders = []

    lexes_not_good = []
    lexes_wo_sorted_triples = []
    lexes_w_wrong_sorted_triples = []

    for l in e.lexes:

        if l['comment'] != 'good':
            lexes_not_good.append(l)

        if not l['sorted_triples']:
            lexes_wo_sorted_triples.append(l)

        if l['comment'] == 'good' and l['sorted_triples']:

            order = tuple(flatten(l['sorted_triples']))
            if len(order) == len(e.triples):
                orders.append(order)
            else:
                lexes_w_wrong_sorted_triples.append(l)

    return (orders,
            lexes_not_good,
            lexes_wo_sorted_triples,
            lexes_w_wrong_sorted_triples)


# para cada entrada do dataset irá analisar as lexes e classificar em 4 grupos
#   lexes com comment != good
#   lexes sem sorted_triples
#   lexes com sorted triples, mas em quantidade inferior ao número de triplas
#   lexes com sorted triples e com quantidade igual ao número de triplas
# não são consideradas entries com número de triplas = 1, visto não
#   haver necessidade de ordenar quando há apenas uma tripla
def make_database(db_name, n_min=2):

    db = load_dataset(db_name)

    dp_db = []

    n_lexes_not_good = 0
    all_lexes_wo_sorted_triples = []
    all_lexes_w_bad_sorted_triples = []

    for e in (e for e in db if len(e.triples) >= n_min):

        (orders,
         lexes_not_good,
         lexes_wo_sorted_triples,
         lexes_w_wrong_sorted_triples) = extract_orders(e)

        if orders:
            dp_db.append((e.triples, orders))

        all_lexes_wo_sorted_triples.extend((e, l) for l
                                           in lexes_wo_sorted_triples)
        all_lexes_w_bad_sorted_triples.extend((e, l) for l
                                              in lexes_w_wrong_sorted_triples)
        n_lexes_not_good += len(lexes_not_good)

    return (dp_db,
            n_lexes_not_good,
            all_lexes_wo_sorted_triples,
            all_lexes_w_bad_sorted_triples)


def acc_same_order_as_triples(dp_db):

    tp = sum(1 for ts, sorteds_ts in dp_db
             if ts in sorteds_ts)

    return tp / len(dp_db)


def acc_random_order(dp_db, seed=0):

    from random import Random

    r = Random(seed)

    tp = sum(1 for ts, sorteds_ts in dp_db
             if tuple(r.sample(ts, len(ts))) in sorteds_ts)

    return tp / len(dp_db)


def acc_majority_order(train_dp_db, test_dp_db):

    from collections import Counter, defaultdict

    order_base = defaultdict(Counter)

    for ts, sorteds_ts in train_dp_db:
        key = tuple(sorted([t.predicate for t in ts]))
        for sorted_ts in sorteds_ts:
            predicate_order = tuple([t.predicate for t in sorted_ts])
            order_base[key][predicate_order] += 1

    order_base = dict(order_base)

    tp = 0

    for ts, sorteds_ts in test_dp_db:
        key = tuple(sorted([t.predicate for t in ts]))

        if key in order_base:
            best_order = order_base[key].most_common()[0][0]

            my_sorted_ts = tuple(sorted(ts,
                                        key=lambda t:
                                            best_order.index(t.predicate)))
        else:
            my_sorted_ts = ts

        if my_sorted_ts in sorteds_ts:
            tp += 1

    return tp / len(test_dp_db)


def acc_ltr(test_dp_db):

    from pretrained_models import ltr_lasso_dp_scorer
    from more_itertools import sort_together
    from itertools import permutations

    m = ltr_lasso_dp_scorer(('train', 'dev'))

    tp = 0

    for ts, sorteds_ts in test_dp_db:

        all_perms = list(permutations(ts))
        scores = m(all_perms, len(ts))
        best_score_perm = sort_together([scores, all_perms],
                                        reverse=True)[1][0]

        if best_score_perm in sorteds_ts:
            tp += 1

    return tp / len(test_dp_db)


def make_markov_scorer(train_dp_db, n=3):

    from nltk.lm import Laplace
    from nltk.util import ngrams
    from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends

    train_texts = [[t.predicate for t in st]
                   for st, sts in train_dp_db for st in sts]

    train_data, padded_texts = padded_everygram_pipeline(n, train_texts)

    model = Laplace(n)
    model.fit(train_data, padded_texts)

    def scorer(triples_list, n_triples=None):

        scores = []

        for triples in triples_list:

            preds = [t.predicate for t in triples]

            score = sum(model.logscore(trigram[-1], trigram[:-1])
                        for trigram in ngrams(pad_both_ends(preds, n=n), n=n)) / len(preds)

            scores.append(score)

        return scores

    return scorer


def acc_markov(scorer, test_dp_db):

    from more_itertools import sort_together
    from itertools import permutations

    tp = 0

    for ts, sorteds_ts in test_dp_db:

        all_perms = list(permutations(ts))
        scores = [scorer(ts_perm) for ts_perm in all_perms]
        best_score_perm = sort_together([scores, all_perms],
                                        reverse=True)[1][0]

        if best_score_perm in sorteds_ts:
            tp += 1

    return tp / len(test_dp_db)


def pct_easy_problems(dp_db):

    from math import factorial

    n_perms = {n: factorial(n) for n in range(1, 8)}

    n_easy = {n: [] for n in range(2, 8)}

    for ts, sorteds_ts in dp_db:

        n_perms_in_ref = len(set(sorteds_ts))

        n_easy[len(ts)].append(n_perms_in_ref / n_perms[len(ts)])

    return n_easy


def evaluate(db_name):

    a = make_database(db_name)[0]
    train_a = make_database('train')[0]
    dev_a = make_database('dev')[0]

    all_train = list(flatten([train_a, dev_a]))

    scorer_2 = make_markov_scorer(all_train, n=2)
    scorer_3 = make_markov_scorer(all_train, n=3)

    subsets = {n: [(ts, sts) for ts, sts in a if len(ts) == n]
               for n in range(2, 8)}

    results = {}
    results['all'] = [
            acc_same_order_as_triples(a),
            acc_random_order(a),
            acc_majority_order(all_train, a),
            acc_ltr(a),
            acc_markov(scorer_2, a),
            acc_markov(scorer_3, a),
            len(a)
            ]

    print('Finished all')

    for n, a_ in subsets.items():

        results[f'{n}'] = [
                acc_same_order_as_triples(a_),
                acc_random_order(a_),
                acc_majority_order(all_train, a_),
                acc_ltr(a_),
                acc_markov(scorer_2, a_),
                acc_markov(scorer_3, a_),
                len(a_)
                ]

        print(f'Finished {n}')

    import pandas as pd

    return pd.DataFrame.from_dict(results,
                                  orient='index',
                                  columns=['same_order',
                                           'random',
                                           'majority',
                                           'ltr',
                                           'markov_n=2',
                                           'markov_n=3',
                                           'len'])


TRIPLE_TEMPLATE = '<{s}, {p}, {o}>'
TRIPLE_SEPARATOR = '#'
TRIPLES_SORTED_TRIPLES_SEPARATOR = '\t'


def serialize_triples(triples):

    triples_repr = (TRIPLE_TEMPLATE.format(s=t.subject,
                                           p=t.predicate,
                                           o=t.object) for t in triples)
    return triples_repr


def serialize_database(dp_db, filepath):

    with open(filepath, 'w', encoding='utf-8') as f:

        f.write('triples\tsorted_triples\n')

        for ts, sorted_ts in dp_db:

            ts_repr = serialize_triples(ts)
            serialized_ts = TRIPLE_SEPARATOR.join(ts_repr)

            sorted_ts_repr = serialize_triples(sorted_ts)
            serialized_sorted_ts = TRIPLE_SEPARATOR.join(sorted_ts_repr)

            f.write('{ts}{sep}{sorted_ts}\n'.format(
                    ts=serialized_ts,
                    sep=TRIPLES_SORTED_TRIPLES_SEPARATOR,
                    sorted_ts=serialized_sorted_ts))
