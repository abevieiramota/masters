# -*- coding: utf-8 -*-
from nltk import sent_tokenize

LEX_TO_USE = 0


# receives 1 triple, return texts for it from 1 triples
def find_1triple(t, db):

    result = []

    t = (t,)

    for e_ in db:

        if t == e_.triples:

            result.append(e_)

    return result


# receives 1 triple, return texts from >1 triples
def find_gt1triples(t, db):

    t = (t,)

    for e_ in db:

        if len(e_.triples) > 1:

            for l in e_.lexes:

                if l['sorted_triples']:

                    if t in l['sorted_triples']:

                        sents = sent_tokenize(l['text'])

                        if len(sents) == len(l['sorted_triples']):

                            for sent, st in zip(sents, l['sorted_triples']):

                                if st == t:
                                    yield sent


def f(triples, db):

    if len(triples) == 1:

        try:
            return next(find_gt1triples(triples[0], db))
        except StopIteration:
            return ''
    else:

        sents = []
        for t in triples:

            try:
                sent = next(find_1triple(t, db))
            except StopIteration:
                sent = ''

            sents.append(sent)

        return ' '.join(sents)


def contains(triples, db):

    result = []

    triples_set = set(triples)

    for e in db:

        if triples_set.issubset(e.triples):
            result.append(e)

    return result


def contains_a(triples, db):

    result = []

    triples_set = set(abstract_triples(triples))

    for e in db:

        if triples_set.issubset(abstract_triples(e.triples)):
            result.append(e)

    return result


def contains_tdb(triples, tdb):

    a_triples = abstract_triples(triples)
    a_triples = set(a_triples)
    result = []

    for (c, a_t), tems in tdb.items():

        if a_triples.issubset(a_t):
            result.extend(tems)

    return result


def containing_text(text, db):

    result = []

    for e in db:

        for l in e.lexes:

            if text in l['text'].lower():

                result.append(l['text'])

    return result
