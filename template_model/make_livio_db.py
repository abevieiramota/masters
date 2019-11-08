# -*- coding: utf-8 -*-
from reading_thiagos_templates import Triple
from more_itertools import flatten


def format_triples(ts):

    return '|'.join('{},{},{}'.format(t.subject,
                                      t.predicate,
                                      t.object) for t in ts)


def f(l):

    refs = {}
    for r in l['references']:
        refs[r['entity']] = r['tag']

    ts = list(flatten(l['sorted_triples']))

    ts_a = [Triple(refs[t.subject],
                   t.predicate,
                   refs[t.object])
            for t in ts]

    tx = l['text']
    te = l['template']

    return '\t'.join([format_triples(ts), format_triples(ts_a), tx, te])


def make(db, filepath):

    with open(filepath, 'w', encoding='utf-8') as fi:
        fi.write('t1\tt2\te1\te2\n')

        for e in db:
            for l in [l for l in e.lexes if l['references'] and l['sorted_triples']]:
                try:
                    data = f(l)
                    fi.write(f'{data}\n')
                except Exception:
                    pass
