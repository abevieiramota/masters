# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, sort_together, flatten
from template_based import abstract_triples
from functools import partial, reduce
from operator import mul
from pretrained_models import (
        load_referrer,
        load_template_selection_lm,
        load_text_selection_lm,
        load_template_db,
        load_discourse_planning,
        load_sentence_aggregation,
        load_template_fallback,
        text_to_id
)
from reading_thiagos_templates import Entry
from random import Random
import logging
import re
import sys
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu
from unidecode import unidecode
from util import top_combinations


TOKENIZER_RE = re.compile(r'(\W)')
def normalize_text(text):

    lex_detokenised = ' '.join(TOKENIZER_RE.split(text))
    lex_detokenised = ' '.join(lex_detokenised.split())

    return unidecode(lex_detokenised.lower())



RE_SPLIT_DOT_COMMA = re.compile(r'([\.,\'])')

def preprocess_text(t):

    return ' '.join(' '.join(RE_SPLIT_DOT_COMMA.split(t)).split())


class TextGenerationPipeline:

    def __init__(self,
                 template_db,
                 tems_lm,
                 txs_lm,
                 dp_scorer,
                 sa_scorer,
                 max_dp,
                 max_sa,
                 max_tems,
                 max_refs,
                 fallback_template,
                 reg,
                 lp_n,
                 lp_a):

        self.template_db = template_db
        self.tems_lm = tems_lm
        self.txs_lm = txs_lm
        self.dp_scorer = dp_scorer
        self.sa_scorer = sa_scorer
        self.max_dp = max_dp
        self.max_sa = max_sa
        self.max_tems = max_tems
        self.max_refs = max_refs
        self.fallback_template = fallback_template
        self.reg = reg
        self.lp_n = lp_n
        self.lp_a = lp_a
        self.logger = logging.getLogger('TextGeneratorPipeline')

    def select_dp(self, entry):

        dps = list(permutations(entry.triples))
        dps_scores = self.dp_scorer(dps)
        dps_scores, dps = sort_together([dps_scores, dps], reverse=True)

        self.logger.debug('Discourse Planning: {}'.format('\n'.join(f'{score:.3f} -> {dp}' for score, dp in zip(dps_scores, dps))))

        return dps[:self.max_dp]

    def select_sa(self, dp):

        sas = list(partitions(dp))
        sas_scores = self.sa_scorer(sas)
        sas_scores, sas = sort_together([sas_scores, sas], reverse=True)

        self.logger.debug('Sentence Aggregation: {}'.format('\n'.join(f'{score:.3f} -> {sa}' for score, sa in zip(sas_scores, sas))))

        return [[tuple(sa_part) for sa_part in sa] for sa in sas[:self.max_sa]]

    def score_template(self, t, a):

        aligned_data = t.align(a)
        refs = [text_to_id(aligned_data[slot_name]) for slot_name, _ in t.slots]
        text = t.fill(refs)
        score = self.tems_lm.score(text) / self.length_penalty(text.split())

        self.logger.debug('Template Selection: {:.3f} -> {}'.format(score, text))

        return score

    # https://arxiv.org/pdf/1609.08144.pdf
    def length_penalty(self, tokens):

        return (self.lp_n + len(tokens))**self.lp_a / (self.lp_n + 1)

    def score_text(self, t):

        t = preprocess_text(t)
        score = self.txs_lm.score(t)
        lp = self.length_penalty(t.split())

        return score / lp

    def make_fallback_text(self, entry):

        sents = []
        for triple in entry.triples:
            template = self.fallback_template(triple.predicate)
            s_ref = self.reg.refer(triple.subject, 'slot-0', '0', template, 1)[1][0]
            o_ref = self.reg.refer(triple.object, 'slot-1', '0', template, 1)[1][0]

            sent = template.fill([s_ref, o_ref])
            sents.append(sent)

        return ' '.join(sents)

    def select_templates(self, sa):

        all_ts = []
        all_scores = []

        for sa_part in sa:

            ts = self.template_db.select(sa_part)

            if not ts:
                return []

            scores = [self.score_template(t, sa_part) for t in ts]
            scores, ts = sort_together([scores, ts], reverse=True)

            all_ts.append(ts[:self.max_tems])
            all_scores.append(scores[:self.max_tems])

        top_combs = top_combinations(all_scores, self.max_refs)

        return [[all_ts[i][ix_] for i, ix_ in enumerate(ix)] for ix in top_combs]
        
    def select_references(self, template, triples):

        aligned_data = template.align(triples)
        # retorna um map de slot_name -> so
        all_refs = []
        all_scores = []
        # contém os identificadores de slots -> usado para dar os replaces no template text

        for slot_name, slot_pos in template.slots:
            so = aligned_data[slot_name]
            # dado que encaixa no slot?
            scores, refs = self.reg.refer(so, slot_name, slot_pos, template, self.max_refs)
            # conjunto de referências encontradas para o dado (so) dado o contexto (ctx) e limitado a uma quantidade (self.max_refs)
            all_refs.append(refs)
            all_scores.append(scores)

        top_combs = top_combinations(all_scores, self.max_refs)

        return [[all_refs[i][ix_] for i, ix_ in enumerate(ix)] for ix in top_combs]

    def make_text(self, entry):

        best_text = None
        best_score = float('-inf')

        for dp in self.select_dp(entry)[:self.max_dp]:

            for sa in self.select_sa(dp)[:self.max_sa]:
                for ts in self.select_templates(sa):
                    # combinação de templates para um particionamento
                    all_sent_texts = []

                    for sa_part, template in zip(sa, ts):
                        # pares de parte e template

                        sent_texts = []
                        # sentenças que dá para construir com o template e as referências

                        for refs in self.select_references(template, sa_part):

                            sent_text = template.fill(refs)
                            sent_texts.append(sent_text)

                        all_sent_texts.append(sent_texts)

                    for sents in product(*all_sent_texts):

                        generated_text = ' '.join(sents)
                        generated_score = self.score_text(generated_text.lower())

                        self.logger.debug(f'Text Selection: {generated_score:.3f} -> {generated_text}')

                        if generated_score > best_score:
                            best_score = generated_score
                            best_text = generated_text

        if not best_text:
            best_text = self.make_fallback_text(entry)

        return best_text


def make_model(params, train_set):
    # 1. Grid Search
    # 1.1. Language Models
    # 1.1.1 Template Selection Language Model
    tems_lm = load_template_selection_lm(train_set, params['tems_lm_n'], params['tems_lm_name'])
    # 1.1.2 Text Selection Language Model
    txs_lm = load_text_selection_lm(train_set, params['txs_lm_n'], params['txs_lm_name'])
    # 1.2 Template database
    fallback_template = load_template_fallback(train_set, params['fallback_template'])
    ns = params.get('tdb_ns', None)
    template_db = load_template_db(train_set, fallback_template=fallback_template, ns=ns)
    # 1.3 Referring Expression Generation
    reg = load_referrer(train_set, params['referrer'], params['referrer_lm_n'])
    # 1.4 Discourse Planning
    dp_scorer = load_discourse_planning(train_set, params['dp_scorer'], params['dp_scorer_n'])
    # 1.5 Sentence Aggregation
    sa_scorer = load_sentence_aggregation(train_set, params['sa_scorer'], params['sa_scorer_n'])
    # 1.6 Template Fallback
    

    # Model
    tgp = TextGenerationPipeline(
            template_db,
            tems_lm,
            txs_lm,
            dp_scorer,
            sa_scorer,
            params['max_dp'],
            params['max_sa'],
            params['max_tems'],
            params['max_refs'],
            fallback_template,
            reg,
            params['lp_n'],
            params['lp_a'])

    return tgp
