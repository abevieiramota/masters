# -*- coding: utf-8 -*-
from util import preprocess_so
from random import Random
from collections import Counter


class EmptyREGer:

    def refer(self, s, slot_pos, slot_type, ctx):

        return ''


class FirstNameOthersPronounREG:

    def __init__(self, ref_db, fallback=preprocess_so):
        self.ref_db = ref_db
        self.fallback = fallback

    def refer(self, s, ctx, max_refs):

        refs_N = self.ref_db['N'].get(s, Counter())
        refs_P = self.ref_db['P'].get(s, Counter())

        refs_N = [v for v, n in refs_N.most_common()]
        refs_P = [v for v, n in refs_P.most_common()]

        if s in ctx['seen']:
            if not refs_P:
                return [self.fallback(s)]
            return refs_P[:max_refs]
        else:
            ctx['seen'].add(s)
            if not refs_N:
                return [self.fallback(s)]
            return refs_N[:max_refs]


class REGer:

    def __init__(self,
                 ref_db,
                 name_db_position=0,
                 pronoun_db_position=0,
                 description_db_position=0,
                 demonstrative_db_position=0,
                 fallback=preprocess_so,
                 is_random=False,
                 seed=None):
        self.ref_db = ref_db
        self.fallback = fallback
        self.positions = {
                'N': name_db_position,
                'P': pronoun_db_position,
                'D': description_db_position,
                'E': demonstrative_db_position
        }
        self.is_random = is_random
        if is_random:
            self.random = Random(seed)

    def get_references(self, s):

        return {k: self.get_reference(s, k)
                for k in ['N', 'P', 'D', 'E']}

    def get_random_reference(self, s):

        refs = set()
        for k in ['N', 'P', 'D', 'E']:
            if s in self.ref_db[k]:
                refs_k = self.ref_db[k][s].keys()
                refs.update(refs_k)
        if refs:
            return self.random.choice(list(refs))
        else:
            return None

    def get_reference(self, s, slot_type):

        if s in self.ref_db[slot_type]:
            counter = self.ref_db[slot_type][s]
            position = self.positions[slot_type]
            return counter.most_common()[position][0]
        else:
            return None

    def refer(self, s, slot_pos, slot_type, ctx):

        if not self.is_random:

            if s in ctx['seen'] or slot_pos > 0:
                if slot_type in ('N', 'D'):
                    slot_type = 'P'
            else:
                if slot_type in ('P', 'D'):
                    slot_type = 'N'

            ctx['seen'].add(s)

            refs = self.get_references(s)

            if slot_type == 'N':
                refs_ordered = [refs[k] for k in ['N', 'D']]

            if slot_type == 'P':
                refs_ordered = [refs[k] for k in ['P', 'D', 'N']]

            if slot_type == 'D':
                refs_ordered = [refs[k] for k in ['D', 'N']]

            if slot_type == 'E':
                refs_ordered = [refs[k] for k in ['E']]

            first_good_ref = None
            for r in refs_ordered:
                if r:
                    first_good_ref = r
                    break
        else:
            first_good_ref = self.get_random_reference(s)

        if first_good_ref:
            return first_good_ref
        else:
            return self.fallback(s)
