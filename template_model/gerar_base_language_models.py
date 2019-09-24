# -*- coding: utf-8 -*-
def make_ref_texts_lm(entries):

    refs = [l['text'].lower() for e in entries for l in e.lexes]

    with open('../data/kenlm/refs_texts.txt', 'w', encoding='utf-8') as f:
        for ref in refs:
            f.write(f'{ref}\n')


def make_ts_texts_lm(outpath):
    # SEE: reading_thiagos_templates.py > make_template_db
    pass
