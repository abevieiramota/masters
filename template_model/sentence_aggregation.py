# -*- coding: utf-8 -*-
from reading_thiagos_templates import load_dataset
from more_itertools import flatten
from collections import defaultdict


# same idea as discourse_planning.extract_aggs
def extract_aggs(e):

    aggs = defaultdict(list)

    lexes_not_good = []
    lexes_wo_sorted_triples = []
    lexes_w_wrong_sorted_triples = []

    for l in e.lexes:

        if l['comment'] != 'good':
            lexes_not_good.append(l)

        if not l['sorted_triples']:
            lexes_wo_sorted_triples.append(l)

        if l['comment'] == 'good' and l['sorted_triples']:

            agg = l['sorted_triples']
            flattened_agg = tuple(list(flatten(agg)))
            if len(flattened_agg) == len(e.triples):
                aggs[flattened_agg].append(agg)
            else:
                lexes_w_wrong_sorted_triples.append(l)

    return (aggs.items(),
            lexes_not_good,
            lexes_wo_sorted_triples,
            lexes_w_wrong_sorted_triples)


def make_database(db_name, n_min=2):

    db = load_dataset(db_name)

    sa_db = []

    all_lexes_not_good = []
    all_lexes_wo_sorted_triples = []
    all_lexes_w_bad_sorted_triples = []

    for e in (e for e in db if len(e.triples) >= n_min):

        (aggs,
         lexes_not_good,
         lexes_wo_sorted_triples,
         lexes_w_wrong_sorted_triples) = extract_aggs(e)

        sa_db.extend(aggs)

        all_lexes_wo_sorted_triples.extend((e, l) for l
                                           in lexes_wo_sorted_triples)
        all_lexes_w_bad_sorted_triples.extend((e, l) for l
                                              in lexes_w_wrong_sorted_triples)
        all_lexes_not_good.extend((e, l) for l
                                  in lexes_not_good)

    return (sa_db,
            all_lexes_not_good,
            all_lexes_wo_sorted_triples,
            all_lexes_w_bad_sorted_triples)


def extract_agg_pattern(agg):

    lens = [len(agg_part) for agg_part in agg]

    return '>'.join([str(l) for l in lens])


def analyze_agg_patterns(sa_db):

    from collections import Counter, defaultdict

    counters = defaultdict(Counter)

    for _, aggs in sa_db:
        for agg in aggs:
            pattern = extract_agg_pattern(agg)
            n = sum([len(agg_part) for agg_part in agg])
            counters[n][pattern] += 1

    return counters
#for n, c_ in c.items():
#    if n <= 5:
#        print(f'n={n} total={sum(c_.values())}')
#        for i, v in c_.most_common():
#            print(f'{i:10}{v}')
#   print()
#
#for n, c_ in c.items():
#    if n > 5:
#        print(f'n={n} total={sum(c_.values())}')
#        for i, v in c_.most_common():
#            print(f'{i:10}{v}')


def acc_all_one_sentence(sa_db):

    tp = sum(1 for ts, aggs in sa_db
             if [ts] in aggs)

    return tp / len(sa_db)


def acc_each_one_sentence(sa_db):

    tp = sum(1 for ts, aggs in sa_db
             if [(t,) for t in ts] in aggs)

    return tp / len(sa_db)


def acc_random(sa_db, seed=0):

    from random import Random
    from more_itertools import partitions

    r = Random(seed)

    tp = 0

    for ts, aggs in sa_db:

        all_partitions = list(partitions(ts))
        random_partition = r.choice(all_partitions)
        random_partition = [tuple(part) for part in random_partition]

        if random_partition in aggs:
            tp += 1

    return tp / len(sa_db)


def acc_majority_agg(train_sa_db, test_sa_db):

    from more_itertools import partitions
    from operator import itemgetter

    counters = analyze_agg_patterns(train_sa_db)

    tp = 0

    for ts, aggs in test_sa_db:

        n = len(ts)

        all_partitions = list(partitions(ts))
        all_patterns = [extract_agg_pattern(agg) for agg in all_partitions]
        all_counts = [counters[n][pat] for pat in all_patterns]

        choosen_agg = max(zip(all_partitions, all_counts),
                          key=itemgetter(1))[0]
        choosen_agg = [tuple(agg_part) for agg_part in choosen_agg]

        if choosen_agg in aggs:
            tp += 1

    return tp / len(test_sa_db)


def evaluate():

    test_a = make_database('test_seen')[0]
    train_a = make_database('train')[0]
    dev_a = make_database('dev')[0]

    all_train = list(flatten([train_a, dev_a]))

    results = {}
    results['all'] = [
            acc_all_one_sentence(test_a),
            acc_each_one_sentence(test_a),
            acc_random(test_a),
            acc_majority_agg(all_train, test_a),
            len(test_a)
            ]

    print('Finished all')

    subsets = {n: [(ts, aggs) for ts, aggs in test_a if len(ts) == n]
               for n in range(2, 8)}

    for n, a_ in subsets.items():

        results[f'{n}'] = [
            acc_all_one_sentence(a_),
            acc_each_one_sentence(a_),
            acc_random(a_),
            acc_majority_agg(all_train, a_),
            len(a_)
            ]

        print(f'Finished {n}')

    import pandas as pd

    return pd.DataFrame.from_dict(results,
                                  orient='index',
                                  columns=['all_in_one_sent',
                                           'each_in_one_sent',
                                           'random',
                                           'majority',
                                           'len'])
