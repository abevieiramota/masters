# -*- coding: utf-8 -*-
from util import preprocess_so
from random import Random
from collections import Counter
from functools import partial
import re


RE_IS_NUMBER = re.compile(r'^[\d\.,]+$')


class EmptyREGer:

    def refer(self, s, slot_pos, slot_type, ctx):

        return ''


class FirstNameOthersPronounREG:

    def __init__(self, ref_db, ref_lm, fallback=preprocess_so, to_print=False):
        self.ref_db = ref_db
        self.fallback = fallback
        self.score_ref = partial(ref_lm.score,
                                 bos=False,
                                 eos=False)
        self.to_print = to_print

    def refer(self, s, ctx, max_refs):

        refs_1st = self.ref_db['1st'].get(s, set())
        refs_1st.add(self.fallback(s).lower())
        refs_2nd = self.ref_db['2nd'].get(s, set())

        slot = '{{{}}}'.format(ctx['slot'])

        if RE_IS_NUMBER.match(s):
            refs_1st.add(s)
            refs_2nd.add(s)

        def score_reg(r):

            text = ctx['t'].template_text.replace(slot, r.replace(' ', '_').lower())

            score = self.score_ref(text)

            return score

        if s not in ctx['seen']:
            # sort by ref_lm
            sorted_refs = sorted(refs_1st, key=score_reg, reverse=True)

            return sorted_refs[:max_refs]
        else:
            if not refs_2nd:
                sorted_refs = sorted(refs_1st, key=score_reg, reverse=True)
                return sorted_refs[:max_refs]
            # sort by ref_lm
            sorted_refs = sorted(refs_2nd, key=score_reg, reverse=True)

            return sorted_refs[:max_refs]
