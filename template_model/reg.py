# -*- coding: utf-8 -*-
from preprocessing import *
from random import Random
from collections import Counter
from functools import partial
from more_itertools import sort_together
import re
import logging
from unidecode import unidecode


class PreprocessREG:
    def __init__(self):
        self.logger = logging.getLogger('PreprocessREG')

    def refer(self, so, slot_name, slot_pos, template, max_refs):
        self.logger.debug(f'{so}')
        return [1.0], [preprocess_so(so)]


class EmptyREGer:
    def refer(self, so, slot_name, slot_pos, template, max_refs):
        return [1.0], ['']


class FirstSecondREG:

    def __init__(self, ref_db, ref_lm):
        self.ref_db = ref_db
        self.ref_lm = ref_lm
        self.logger = logging.getLogger('FirstNameOthersPronounREG')
    
    def fallback(self, so):

        return normalize_text(preprocess_so(so))

    def refer(self, so, slot_name, slot_pos, template, max_refs):

        refs_1st = self.ref_db['1st'].get(so, set())
        refs_1st.add(self.fallback(so))

        refs_2nd = self.ref_db['2nd'].get(so, set())

        slot = '{{{}}}'.format(f'{slot_name}-{slot_pos}')

        def score_reg(r):

            ref_id = text_to_id(r)
            text = template.template_text.replace(slot, ref_id)
            score = self.ref_lm.score(text)

            self.logger.debug(f'{score:.3f} -> {ref_id} -> {text}')

            return score

        if slot_pos == '0' or not refs_2nd:
            refs = refs_1st 
        else:
            refs = refs_2nd

        #refs = refs_1st | refs_2nd

        scores = [score_reg(r) for r in refs]
        scores, sorted_refs = sort_together([scores, refs], reverse=True)

        return scores[:max_refs], sorted_refs[:max_refs]
