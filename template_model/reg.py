# -*- coding: utf-8 -*-
from reading_thiagos_templates import preprocess_so


class REGer:

    def __init__(self,
                 name_db,
                 pronoun_db,
                 name_db_position=0,
                 pronoun_db_position=0,
                 fallback_name=lambda t: preprocess_so(t),
                 fallback_pronoun=lambda t: ' '):
        self.name_db = name_db
        self.pronoun_db = pronoun_db
        self.fallback_name = fallback_name
        self.fallback_pronoun = fallback_pronoun
        self.name_db_position = name_db_position
        self.pronoun_db_position = pronoun_db_position

    def refer(self, s, ctx):

        if s in ctx['seen']:

            if s in self.pronoun_db:

                return self.pronoun_db[s].most_common()[self.pronoun_db_position][0]
            else:
                return self.fallback_pronoun(s)

        ctx['seen'].add(s)

        if s in self.name_db:

            return self.name_db[s].most_common()[self.name_db_position][0]
        else:
            return self.fallback_name(s)
