# -*- coding: utf-8 -*-
from util import preprocess_so
from random import Random
from collections import Counter
from functools import partial
from more_itertools import sort_together
import re


RE_IS_NUMBER = re.compile(r'^[\d\.,]+$')


class EmptyREGer:

    def refer(self, s, slot_pos, slot_type, ctx):

        return ''


class FirstNameOthersPronounREG:

    def __init__(self, ref_db, ref_lm, fallback=preprocess_so):
        self.ref_db = ref_db
        self.fallback = fallback
        self.score_ref = ref_lm.score

    def refer(self, s, ctx, max_refs):

        refs_1st = self.ref_db['1st'].get(s, set())
        refs_1st.add(self.fallback(s).lower())
        refs_2nd = self.ref_db['2nd'].get(s, set())

        slot = '{{{}}}'.format(ctx['slot'])

        if RE_IS_NUMBER.match(s):
            refs_1st.add(s)
            refs_2nd.add(s)

        def score_reg(r):

            text = ctx['t'].template_text.replace(slot, r.replace(' ', '_'))

            score = self.score_ref(text)
            # print(f'REG: {score}\n\t{text}')

            return score

        if s not in ctx['seen']:
            scores = [score_reg(r) for r in refs_1st]
            scores, sorted_refs = sort_together([scores, refs_1st], reverse=True)

            return scores[:max_refs], sorted_refs[:max_refs]
        else:
            if not refs_2nd:
                scores = [score_reg(r) for r in refs_1st]
                scores, sorted_refs = sort_together([scores, refs_1st], reverse=True)

                return scores[:max_refs], sorted_refs[:max_refs]

            scores = [score_reg(r) for r in refs_2nd]
            scores, sorted_refs = sort_together([scores, refs_2nd], reverse=True)

            return scores[:max_refs], sorted_refs[:max_refs]
