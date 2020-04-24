# -*- coding: utf-8 -*-
from more_itertools import flatten
from heapq import heappush, heappop


def clear_dir(dirpath):

    import glob
    import os

    files = glob.glob(f'{dirpath}/*')
    for f in files:
        os.remove(f)


def extract_orders(e):

    orders = []

    for l in [l for l in e.lexes if l['comment'] == 'good'
              and l['sorted_triples']]:

        order = tuple(flatten(l['sorted_triples']))
        if len(order) == len(e.triples):
            orders.append(order)
        else:
            orders.append(None)

    return orders


def top_combinations(X, n):

    explored = []
    n_lists = len(X)
    lists_size = [len(x) for x in X]

    X = [sorted(x) for x in X]

    heappush(explored, (sum(x[0] for x in X), [0]*n_lists))

    result = []
    while len(result) < n:

        if len(explored) == 0:
            break

        _, ixs = heappop(explored)

        result.append(ixs)

        for i in range(0, n_lists):
            if len(result) == n:
                break 

            if ixs[i] == lists_size[i] - 1:
                continue

            ixs_copy = ixs[::]
            ixs_copy[i] = ixs_copy[i] + 1

            heappush(explored, (sum(x[ixs_] for ixs_, x in zip(ixs_copy, X)), ixs_copy))

    return result