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
        load_template_fallback
)
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from reading_thiagos_templates import Entry
from random import Random
import re
import sys
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu


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

    def select_dp(self, entry):

        dps = list(permutations(entry.triples))
        dps_scores = self.dp_scorer(dps)
        dps = sort_together([dps_scores, dps],
                            reverse=True)[1]

        return dps

    def select_sa(self, dp):

        sas = list(partitions(dp))
        sas_scores = self.sa_scorer(sas)
        sas = sort_together([sas_scores, sas],
                            reverse=True)[1]

        return sas

    def score_template(self, t, a):

        aligned_data = t.align(a)
        reg_data = {}
        for slot_name, slot_pos in t.slots:
            reg_data[(f'{slot_name}-{slot_pos}')] = aligned_data[slot_name]
        text = t.fill(reg_data, a)
        score = self.tems_lm.score(text)
        lp = self.length_penalty(text.split())

        return score / lp

    # https://arxiv.org/pdf/1609.08144.pdf
    def length_penalty(self, tokens):

        return (self.lp_n + len(tokens))**self.lp_a / (self.lp_n + 1)

    def score_text(self, t):

        t = preprocess_text(t)
        score = self.txs_lm.score(t)
        lp = self.length_penalty(t.split())

        return score / lp

    def make_fallback_text(self, entry):

        ctx = {'seen': set()}
        sent_texts = [self.fallback_template.fill([a], self.reg, ctx)
                      for a in entry.triples]
        return ' '.join(sent_texts)

    def select_templates(self, entry, sa):

        all_ts = []
        all_scores = []

        for sa_part in sa:

            ts = self.template_db.select(entry.category, sa_part)

            if not ts:
                return []

            scores = [self.score_template(t, sa_part) for t in ts]
            scores, ts = sort_together([scores, ts], reverse=True)

            all_ts.append(ts[:self.max_tems])
            all_scores.append(scores[:self.max_tems])
        
        all_ts_comb = list(product(*all_ts))
        all_scores_comb = [sum(scores) for scores in product(*all_scores)]

        selected_templates = sort_together([all_scores_comb, all_ts_comb], reverse=True)[1][:self.max_tems]

        return selected_templates

    def make_text(self, entry):

        m_t_n = {t:i for i, t in enumerate(entry.triples)}
        for t, i in m_t_n.items():
            print(f'{i} -> {t}')

        best_text = None
        best_score = float('-inf')
        n_dp_used = 0

        for dp in self.select_dp(entry):
            print(f'Order: ', [m_t_n[t] for t in dp])
            if n_dp_used == self.max_dp:
                break

            n_sa_used = 0
            for sa in self.select_sa(dp):
                print(f'Agg: ', [[m_t_n[t] for t in part] for part in sa])
                if n_sa_used == self.max_sa:
                    break 

                templates = self.select_templates(entry, sa)
                for tems in templates:
                    for sa_part, t in zip(sa, tems):
                        print([m_t_n[t] for t in sa_part],  f'-> {t.template_text}')
                    print()
                if not templates:
                    print('>>>>Agg not used')
                    continue
                
                n_sa_used += 1
                
                for ts in templates:
                    # combinação de templates para um particionamento

                    ctx = {'seen': set()}

                    all_sent_texts = []

                    for a, t in zip(sa, ts):
                        # pares de parte e template

                        sent_texts = []
                        # sentenças que dá para construir com o template e as referências

                        aligned_data = t.align(a)
                        # ??? o que é Template.align?
                        all_refs = []
                        all_scores = []
                        slots = []
                        # contém os identificadores de slots -> usado para dar os replaces no template text

                        ctx['t'] = t

                        for slot_name, slot_pos in t.slots:
                            so = aligned_data[slot_name]
                            # dado que encaixa no slot?
                            slot = f'{slot_name}-{slot_pos}'
                            # identificar do slot?
                            ctx['slot'] = slot
                            scores, refs = self.reg.refer(so, ctx, self.max_refs)
                            # conjunto de referências encontradas para o dado (so) dado o contexto (ctx) e limitado a uma quantidade (self.max_refs)
                            slots.append(slot)
                            all_refs.append(refs)
                            all_scores.append(scores)

                        all_refs_comb = list(product(*all_refs))
                        all_scores_comb = [sum(scores) for scores in product(*all_scores)]

                        selected_refs = sort_together([all_scores_comb, all_refs_comb], reverse=True)[1][:self.max_refs]
                        
                        for a_t in a:
                            ctx['seen'].add(a_t.subject)
                            ctx['seen'].add(a_t.object)

                        for refs in selected_refs:

                            reg_data = {}
                            for slot, ref in zip(slots, refs):
                                reg_data[slot] = ref

                            sent_text = t.fill(reg_data, a)
                            sent_texts.append(sent_text)

                        all_sent_texts.append(sent_texts)

                    for sents in product(*all_sent_texts):

                        generated_text = ' '.join(sents)
                        if generated_text[-1] != '.':
                            generated_text = f'{generated_text} .'
                        generated_score = self.score_text(generated_text.lower())

                        if generated_score > best_score:
                            best_score = generated_score
                            best_text = generated_text

            if n_sa_used > 0:
                n_dp_used += 1

        if not best_text:
            best_text = self.make_fallback_text(entry)

        return best_text


def make_model(params, train_set):
    # 1. Grid Search
    # 1.1. Language Models
    # 1.1.1 Template Selection Language Model
    tems_lm = load_template_selection_lm(train_set,
                                         params['tems_lm_n'],
                                         params['tems_lm_name'])
    # 1.1.2 Text Selection Language Model
    txs_lm = load_text_selection_lm(train_set,
                                    params['txs_lm_n'],
                                    params['txs_lm_name'])
    # 1.2 Template database
    fallback_template = load_template_fallback(train_set, params['fallback_template'])
    ns = params.get('tdb_ns', None)
    template_db = load_template_db(train_set, fallback_template=fallback_template, ns=ns)
    # 1.3 Referring Expression Generation
    reg = load_referrer(train_set,
                        params['referrer'],
                        params['referrer_lm_n'])
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
