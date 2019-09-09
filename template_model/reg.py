# -*- coding: utf-8 -*-
from util import preprocess_so
import pickle


def load_pronoun_db():

    with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
        pronoun_db = pickle.load(f)

    return pronoun_db


def load_name_db():

    with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
        name_db = pickle.load(f)

    return name_db


class REGer:

    def __init__(self, pronoun_db, name_db):
        self.pronoun_db = pronoun_db
        self.name_db = name_db

    def refer(self, s, ctx):

        if s in ctx['seen']:

            if s in self.pronoun_db:

                return self.pronoun_db[s].most_common()[0][0]
            else:
                return ''

        ctx['seen'].add(s)

        if s in self.name_db:

            return self.name_db[s].most_common()[0][0]
        else:
            return preprocess_so(s)
