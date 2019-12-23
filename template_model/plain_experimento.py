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
        load_preprocessing
)
from gerar_base_sentence_aggregation import SentenceAggregationFeatures
from gerar_base_discourse_planning import DiscoursePlanningFeatures
from reading_thiagos_templates import Entry
import sys
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu
from random import Random


class TextGenerationPipeline:

    def __init__(self,
                 template_db,
                 tems_lm,
                 tems_lm_bos,
                 tems_lm_eos,
                 tems_lm_preprocess_input,
                 txs_lm,
                 txs_lm_bos,
                 txs_lm_eos,
                 txs_lm_preprocess_input,
                 dp_scorer,
                 sa_scorer,
                 max_tems,
                 max_refs,
                 max_texts,
                 fallback_template,
                 reg,
                 lp_n,
                 lp_a):

        self.template_db = template_db
        self.categories = set(c for (c, _) in template_db.keys())
        self.tems_lm = tems_lm
        self.tems_lm_score = partial(tems_lm.score,
                                     bos=tems_lm_bos,
                                     eos=tems_lm_eos)
        self.tems_lm_bos = tems_lm_bos
        self.tems_lm_eos = tems_lm_eos
        self.tems_lm_preprocess_input = tems_lm_preprocess_input
        self.txs_lm = txs_lm
        self.txs_lm_score = partial(txs_lm.score,
                                    bos=txs_lm_bos,
                                    eos=txs_lm_eos)
        self.txs_lm_bos = txs_lm_bos
        self.txs_lm_eos = txs_lm_eos
        self.txs_lm_preprocess_input = txs_lm_preprocess_input
        self.dp_scorer = dp_scorer
        self.sa_scorer = sa_scorer
        self.max_tems = max_tems
        self.max_refs = max_refs
        self.max_texts = max_texts
        self.fallback_template = fallback_template
        self.reg = reg
        self.lp_n = lp_n
        self.lp_a = lp_a
        self.random_tems = Random(345)
        self.random_refs = Random(456)

    def select_discourse_planning(self, entry, n_triples):

        dps = list(permutations(entry.triples))
        dps_scores = self.dp_scorer(dps, n_triples)
        dps = sort_together([dps_scores, dps],
                            reverse=True)[1]

        return dps

    def select_sentence_aggregation(self, dp, n_triples):

        sas = list(partitions(dp))
        sas_scores = self.sa_scorer(sas, n_triples)
        sas = sort_together([sas_scores, sas],
                            reverse=True)[1]

        return sas

    def score_template(self, t, a):

        aligned_data = t.align(a)
        reg_data = {}
        for slot_name, slot_pos in t.slots:
            reg_data[(f'{slot_name}-{slot_pos}')] = aligned_data[slot_name]
        text = t.fill(reg_data, a)
        preprocessed_text = self.tems_lm_preprocess_input(text)

        score = self.tems_lm_score(preprocessed_text)

        return score

    # https://arxiv.org/pdf/1609.08144.pdf
    def length_penalty(self, tokens):

        return (self.lp_n + len(tokens))**self.lp_a / (self.lp_n + 1)

    def score_text(self, t):

        preprocessed_text = self.txs_lm_preprocess_input(t)

        score = self.txs_lm_score(preprocessed_text)
        lp = self.length_penalty(t.split())

        return score / lp

    def make_fallback_text(self, entry):

        ctx = {'seen': set()}
        sent_texts = [self.fallback_template.fill([a], self.reg, ctx)
                      for a in entry.triples]
        return ' '.join(sent_texts)

    def select_templates(self, entry, sa):

        all_ts = []

        for sa_part in sa:

            a_triples = abstract_triples(sa_part)
            c_key = (entry.category, a_triples)

            if c_key in self.template_db:
                ts = self.template_db[c_key]
            else:
                ts = list(flatten(self.template_db.get((c, a_triples), [])
                                  for c in self.categories))

            if ts:
                sorted_ts = sorted(ts,
                                   key=lambda t: self.score_template(t, sa_part),
                                   reverse=True)
                all_ts.append(sorted_ts[:self.max_tems])
            elif len(sa_part) == 1:
                all_ts.append([self.fallback_template])
            else:
                return []

        return product(*all_ts)

    def make_text(self, entry):

        best_text = None
        best_score = float('-inf')
        n_triples = len(entry.triples)
        n_texts = 0

#        b = time.time()
        for dp in self.select_discourse_planning(entry, n_triples):
#            print(time.time() - b)
#            b = time.time()
            for sa in self.select_sentence_aggregation(dp, n_triples):
#                print(time.time() - b)
                for ts in self.select_templates(entry, sa):

                    ctx = {'seen': set()}

                    all_sent_texts = []

                    for a, t in zip(sa, ts):

                        sent_texts = []

                        aligned_data = t.align(a)
                        all_refs = []
                        slots = []

                        ctx['t'] = t

                        for slot_name, slot_pos in t.slots:
                            so = aligned_data[slot_name]
                            slot = f'{slot_name}-{slot_pos}'
                            ctx['slot'] = slot
                            refs = self.reg.refer(so, ctx, self.max_refs)
                            slots.append(slot)
                            all_refs.append(refs)

                        for a_t in a:
                            ctx['seen'].add(a_t.subject)
                            ctx['seen'].add(a_t.object)

                        for refs in product(*all_refs):

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
                        n_texts += 1
                        generated_score = self.score_text(generated_text)

#                        print(f'{generated_text}\n{generated_score}\n')

                        if generated_score > best_score:
                            best_score = generated_score
                            best_text = generated_text

                        if n_texts >= self.max_texts:
                            return best_text

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
    tems_lm_preprocess_input = load_preprocessing(
            params['tems_lm_preprocess_input'])
    # 1.1.2 Text Selection Language Model
    txs_lm = load_text_selection_lm(train_set,
                                    params['txs_lm_n'],
                                    params['txs_lm_name'])
    txs_lm_preprocess_input = load_preprocessing(
            params['txs_lm_preprocess_input'])
    # 1.2 Template database
    template_db = load_template_db(train_set, params.get('tdb_ns', None))
    # 1.3 Referring Expression Generation
    reg = load_referrer(train_set,
                        params['referrer'],
                        params['referrer_lm_n'])
    # 1.4 Discourse Planning
    dp_scorer = load_discourse_planning(train_set, params['dp_scorer'])
    # 1.5 Sentence Aggregation
    sa_scorer = load_sentence_aggregation(train_set, params['sa_scorer'])
    # 1.6 Template Fallback
    fallback_template = load_template_fallback(train_set,
                                               params['fallback_template'])

    # Model
    tgp = TextGenerationPipeline(
            template_db,
            tems_lm,
            params['tems_lm_bos'],
            params['tems_lm_eos'],
            tems_lm_preprocess_input,
            txs_lm,
            params['txs_lm_bos'],
            params['txs_lm_eos'],
            txs_lm_preprocess_input,
            dp_scorer,
            sa_scorer,
            params['max_tems'],
            params['max_refs'],
            params['max_texts'],
            fallback_template,
            reg,
            params['lp_n'],
            params['lp_a'])

    return tgp
