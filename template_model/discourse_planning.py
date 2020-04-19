# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset
from more_itertools import flatten


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


def acc_markov(scorer, test_dp_db):

    from more_itertools import sort_together
    from itertools import permutations

    tp = 0

    for ts, sorteds_ts in test_dp_db:

        all_perms = list(permutations(ts))
        scores = scorer(all_perms)
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


def evaluate():

    a = make_database('test_seen')[0]
    train_a = make_database('train')[0]
    dev_a = make_database('dev')[0]

    all_train = list(flatten([train_a, dev_a]))

    scorer_2 = make_markov_scorer(all_train, n=2)
    scorer_3 = make_markov_scorer(all_train, n=3)
    scorer_4 = make_markov_scorer(all_train, n=4)

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
            acc_markov(scorer_4, a),
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
                acc_markov(scorer_4, a_),
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
                                           'markov_n=4',
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
