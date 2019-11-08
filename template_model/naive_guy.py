# -*- coding: utf-8 -*-
from nltk import sent_tokenize

LEX_TO_USE = 0


# receives 1 triple, return texts for it from 1 triples
def find_1triple(t, db):

    t = (t,)

    for e_ in db:

        if t == e_.triples:

            for l in e_.lexes:

                yield l['text']


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


def f(e, db):

    if len(e.triples) == 1:

        try:
            return next(find_gt1triples(e.triples[0], db))
        except StopIteration:
            return ''
    else:

        sents = []
        for t in e.triples:

            try:
                sent = next(find_1triple(t, db))
            except StopIteration:
                sent = ''

            sents.append(sent)

        return ' '.join(sents)
