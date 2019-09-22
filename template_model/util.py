# -*- coding: utf-8 -*-
from more_itertools import flatten


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
