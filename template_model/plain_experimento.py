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

# https://arxiv.org/pdf/1609.08144.pdf
def length_penalty(tokens, n, a):

    return (n + len(tokens))**a / (n + 1)


NUMERO_GRANDE = 10**5


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
                 lp_tems_n,
                 lp_tems_a,
                 lp_txs_n,
                 lp_txs_a):

        self.template_db = template_db
        self.tems_lm = tems_lm
        self.txs_lm = txs_lm
        self.dp_scorer = dp_scorer
        self.sa_scorer = sa_scorer
        self.max_dp = max_dp if max_dp > 0 else NUMERO_GRANDE
        self.max_sa = max_sa if max_sa > 0 else NUMERO_GRANDE
        self.max_tems = max_tems if max_tems > 0 else NUMERO_GRANDE
        self.max_refs = max_refs if max_refs > 0 else NUMERO_GRANDE
        self.fallback_template = fallback_template
        self.reg = reg
        self.lp_tems_n = lp_tems_n
        self.lp_tems_a = lp_tems_a
        self.lp_txs_n = lp_txs_n
        self.lp_txs_a = lp_txs_a
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
        score = self.tems_lm.score(text) / length_penalty(text.split(), self.lp_tems_a, self.lp_tems_n)

        t_predicates = ' - '.join(t_.predicate for t_ in t.template_triples)
        self.logger.debug('Template Selection - scoring: {:.3f} -> <{}> - {} -> {}'.format(score, t_predicates, t.template_text, text))

        return score

    def score_text(self, t):

        t = preprocess_text(t)
        score = self.txs_lm.score(t) / length_penalty(t.split(), self.lp_txs_a, self.lp_txs_n)

        return score

    def make_fallback_text(self, entry):

        sents = []
        already_seen = set()
        for triple in entry.triples:
            template = self.fallback_template(triple.predicate)
            s_ref = self.reg.refer(triple.subject, 'slot-0', '0', template, 1, triple.subject in already_seen)[1][0]
            already_seen.add(triple.subject)
            o_ref = self.reg.refer(triple.object, 'slot-1', '0', template, 1, triple.object in already_seen)[1][0]
            already_seen.add(triple.object)

            sent = template.fill([s_ref, o_ref])
            sents.append(sent)

        return ' '.join(sents)

    def select_templates(self, sa):

        tss = []

        for sa_part in sa:

            ts = self.template_db.select(sa_part)

            if not ts:
                return []

            tss.append(ts)

        all_ts = []
        all_scores = []

        for ts, sa_part in zip(tss, sa):
            scores = [self.score_template(t, sa_part) for t in ts]
            scores, ts = sort_together([scores, ts], reverse=True)

            all_ts.append(ts[:self.max_tems])
            all_scores.append(scores[:self.max_tems])

        top_combs = top_combinations(all_scores, self.max_refs)
        top_template_combs = [[all_ts[i][ix_] for i, ix_ in enumerate(ix)] for ix in top_combs]

        self.logger.debug('Template Selection - top combs: {}'.format('\n'.join(' ;; '.join(t.template_text for t in tems) for tems in top_template_combs)))

        return top_template_combs
        
    def select_references_with_texts(self, ts, sa):

        all_refs = []
        all_scores = []
        already_seen_ref = set()

        for template, triples in zip(ts, sa):        

            aligned_data = template.align(triples)
            # retorna um map de slot_name -> so

            for slot_name, slot_pos in template.slots:
                so = aligned_data[slot_name]
                # dado que encaixa no slot?
                scores, refs = self.reg.refer(so, slot_name, slot_pos, template, self.max_refs, so in already_seen_ref)
                # conjunto de referências encontradas para o dado (so) dado o contexto (ctx) e limitado a uma quantidade (self.max_refs)
                all_refs.append(refs)
                all_scores.append(scores)
                already_seen_ref.add(so)

        top_combs = top_combinations(all_scores, self.max_refs)
        top_refs = [[all_refs[i][ix_] for i, ix_ in enumerate(ix)] for ix in top_combs]

        top_texts = []

        for top_ref in top_refs:

            sents = []
            ix_ini = 0
            for template in ts:
                ix_fim = ix_ini + len(template.slots)
                sent = template.fill(top_ref[ix_ini:ix_fim])
                sents.append(sent)
                ix_ini = ix_fim
            top_texts.append(' '.join(sents))

        return top_texts

    def make_text(self, entry):

        best_text = None
        best_score = float('-inf')

        for dp in self.select_dp(entry):
            for sa in self.select_sa(dp):
                for ts in self.select_templates(sa):
                    for generated_text in self.select_references_with_texts(ts, sa):

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
    dp_scorer = load_discourse_planning(train_set, params['dp_scorer'], params['dp_lm_n'])
    # 1.5 Sentence Aggregation
    sa_scorer = load_sentence_aggregation(train_set, params['sa_scorer'], params['sa_lm_n'])
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
            params['lp_tems_n'],
            params['lp_tems_a'],
            params['lp_txs_n'],
            params['lp_txs_a'])

    return tgp
