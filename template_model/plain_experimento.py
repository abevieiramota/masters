# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, sort_together
from template_based import abstract_triples
from functools import partial
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
                 max_dp,
                 max_sa,
                 max_tems,
                 fallback_template,
                 referrer):

        self.template_db = template_db
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
        self.max_dp = max_dp
        self.max_sa = max_sa
        self.max_tems = max_tems
        self.fallback_template = fallback_template
        self.referrer = referrer

    def select_discourse_planning(self, entry, n_triples):

        dps = list(permutations(entry.triples))
        dps_scores = self.dp_scorer(dps, n_triples)
        dps = sort_together([dps_scores, dps],
                            reverse=True)[1]

        return dps[:self.max_dp]

    def select_sentence_aggregation(self, dp, n_triples):

        sas = list(partitions(dp))
        sas_scores = self.sa_scorer(sas, n_triples)
        sas = sort_together([sas_scores, sas],
                            reverse=True)[1]

        return sas[:self.max_sa]

    def score_template(self, t, a):

        text = t.fill(a, lambda so, slot_pos, slot_type, ctx: so, None)
        preprocessed_text = self.tems_lm_preprocess_input(text)

        return self.tems_lm_score(preprocessed_text)

    def select_templates(self, ts, a):

        sorted_ts = sorted(ts,
                           key=lambda t: self.score_template(t, a),
                           reverse=True)
        return sorted_ts[:self.max_tems]

    def score_text(self, t):

        preprocesssed_text = self.txs_lm_preprocess_input(t)

        return self.txs_lm_score(preprocesssed_text)

    def make_text(self, entry):

        return self.make_texts(entry)[0]

    def make_texts(self, entry):

        texts = []
        n_triples = len(entry.triples)

        dps = self.select_discourse_planning(entry, n_triples)

        for dp in dps:

            sas = self.select_sentence_aggregation(dp, n_triples)

            for sa in sas:

                templates = []

                for sa_part in sa:

                    a_part = abstract_triples(sa_part)
                    t_key = (entry.category, a_part)

                    if t_key in self.template_db:
                        ts = self.template_db[t_key]

                        selected_ts = self.select_templates(ts, sa_part)

                        templates.append(selected_ts)
                    elif len(a_part) == 1:
                        templates.append([self.fallback_template])
                    else:
                        templates.append([])

                for ts in product(*templates):

                    ctx = {'seen': set()}

                    sent_texts = [t.fill(a, self.referrer, ctx)
                                  for a, t in zip(sa, ts)]
                    templates_info = [(t == self.fallback_template,
                                       len(t.template_triples)) for t in ts]

                    texts.append((' '.join(sent_texts),
                                  templates_info,
                                  n_triples))

        texts = sorted(texts,
                       key=lambda tti: self.score_text(tti[0]),
                       reverse=True)

        return texts


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
    template_db = load_template_db(train_set)
    # 1.3 Referring Expression Generation
    referrer = load_referrer(train_set, params['referrer'])
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
            params['max_dp'],
            params['max_sa'],
            params['max_tems'],
            fallback_template,
            referrer)

    return tgp
