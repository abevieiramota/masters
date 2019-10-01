# -*- coding: utf-8 -*-
from util import preprocess_so


class REGer:

    def __init__(self,
                 ref_db,
                 name_db_position=0,
                 pronoun_db_position=0,
                 description_db_position=0,
                 demonstrative_db_position=0,
                 fallback=preprocess_so):
        self.ref_db = ref_db
        self.fallback = fallback
        self.name_db_position = name_db_position
        self.pronoun_db_position = pronoun_db_position
        self.description_db_position = description_db_position
        self.demonstrative_db_position = demonstrative_db_position

    def refer(self, s, slot_pos, slot_type, ctx):

        if s in ctx['seen'] or slot_pos > 0:

            if slot_type == 'N':
                slot_type = 'P'

        ctx['seen'].add(s)

        if slot_type == 'N':
            if s in self.ref_db['N']:
                return self.ref_db['N'][s].most_common()[self.name_db_position][0]
            else:
                slot_type = 'P'

        if slot_type == 'P':
            if s in self.ref_db['P']:
                return self.ref_db['P'][s].most_common()[self.pronoun_db_position][0]
            else:
                slot_type = 'D'

        if slot_type == 'D':
            if s in self.ref_db['D']:
                return self.ref_db['D'][s].most_common()[self.description_db_position][0]
            else:
                slot_type == 'E'

        if slot_type == 'E':
            if s in self.ref_db['E']:
                return self.ref_db['E'][s].most_common()[self.demonstrative_db_position][0]
        else:
            return self.fallback(s)
