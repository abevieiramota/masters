# -*- coding: utf-8 -*-
from util import load_train_dev, Entry


def make_ref_texts_lm(outpath):

    td = load_train_dev()

    refs = [l['text'].lower() for e in td for l in e.lexes]

    with open(outpath, 'w', encoding='utf-8') as f:
        for ref in refs:
            f.write(f'{ref}\n')


def make_ts_texts_lm(outpath):
    # SEE: notebook 01 - Templates - Thiago's dataset
    pass