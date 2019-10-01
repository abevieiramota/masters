# -*- coding: utf-8 -*-
from more_itertools import flatten
import re


PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')
CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')


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


def preprocess_so(so):

    parenthesis_preprocessed = PARENTHESIS_RE.sub(r'\g<2> \g<1>', so)
    underline_removed = parenthesis_preprocessed.replace('_', ' ')
    camelcase_preprocessed = CAMELCASE_RE.sub(r'\g<1> \g<2>',
                                              underline_removed)

    return camelcase_preprocessed.strip('" ')
