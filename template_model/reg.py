# -*- coding: utf-8 -*-
from reading_thiagos_templates import preprocess_so


class REGer:

    def __init__(self, name_db, pronoun_db):
        self.name_db = name_db
        self.pronoun_db = pronoun_db

    def refer(self, s, ctx):

        if s in ctx['seen']:

            if s in self.pronoun_db:

                return self.pronoun_db[s].most_common()[0][0]
            else:
                return ' '

        ctx['seen'].add(s)

        if s in self.name_db:

            return self.name_db[s].most_common()[0][0]
        else:
            return preprocess_so(s)
