# -*- coding: utf-8 -*-
from util import preprocess_so
from random import Random
from collections import Counter
from functools import partial
from more_itertools import sort_together
import re
import logging
from unidecode import unidecode


TOKENIZER_RE = re.compile(r'(\W)')
def normalize_text(text):

    lex_detokenised = ' '.join(TOKENIZER_RE.split(text))
    lex_detokenised = ' '.join(lex_detokenised.split())

    return unidecode(lex_detokenised.lower())


RE_IS_NUMBER = re.compile(r'^[\d\.,]+$')


class EmptyREGer:

    def refer(self, s, slot_pos, slot_type, ctx):

        return ''


class FirstNameOthersPronounREG:

    def __init__(self, ref_db, ref_lm):
        self.ref_db = ref_db
        self.score_ref = ref_lm.score
        self.logger = logging.getLogger('FirstNameOthersPronounREG')
    
    def fallback(self, so):

        return normalize_text(preprocess_so(so))

    def refer(self, so, slot_name, slot_pos, template, max_refs):

        refs_1st = self.ref_db['1st'].get(so, set())
        refs_1st.add(self.fallback(so))
        refs_2nd = self.ref_db['2nd'].get(so, set())

        slot = '{{{}}}'.format(f'{slot_name}-{slot_pos}')

        if RE_IS_NUMBER.match(so):
            refs_1st.add(so)
            refs_2nd.add(so)

        def score_reg(r):

            text = template.template_text.replace(slot, r.replace(' ', '_'))
            score = self.score_ref(text)

            self.logger.debug(f'{score:.3f} -> {r} -> {text}')

            return score

        if slot_pos == '0':
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
