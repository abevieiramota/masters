# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, sort_together, flatten
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
from reading_thiagos_templates import Entry, load_dataset
from nltk.lm import Laplace
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
import sys
sys.path.append('../evaluation')
from evaluate import normalize_text


class TextGenerationPipeline:

    def __init__(self,
                 db,
                 template_db,
                 tems_lm,
                 tems_lm_bos,
                 tems_lm_eos,
                 tems_lm_preprocess_input,
                 dp_scorer,
                 sa_scorer,
                 max_dp,
                 max_sa,
                 max_tems,
                 max_refs,
                 fallback_template,
                 reg):

        self.db = db
        self.fallback_txs_lm = self.make_all_model()
        self.template_db = template_db
        self.categories = set(c for (c, _) in template_db.keys())
        self.tems_lm = tems_lm
        self.tems_lm_score = partial(tems_lm.score,
                                     bos=tems_lm_bos,
                                     eos=tems_lm_eos)
        self.tems_lm_bos = tems_lm_bos
        self.tems_lm_eos = tems_lm_eos
        self.tems_lm_preprocess_input = tems_lm_preprocess_input
        self.dp_scorer = dp_scorer
        self.sa_scorer = sa_scorer
        self.max_dp = max_dp
        self.max_sa = max_sa
        self.max_tems = max_tems
        self.max_refs = max_refs
        self.fallback_template = fallback_template
        self.reg = reg

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

        #print('{}\n{}\n{}\n'.format(score, n_score, text))

        return score

    def select_templates(self, ts, a):

        sorted_ts = sorted(ts,
                           key=lambda t: self.score_template(t, a),
                           reverse=True)
        return sorted_ts[:self.max_tems]

    def make_all_model(self):

        texts = []

        for e in self.db:
            for l in e.lexes:
                texts.append(l['text'])

        tokenized_texts = [normalize_text(x).split() for x in texts]

        n = 3
        train_data, padded_texts = padded_everygram_pipeline(n, tokenized_texts)

        model = Laplace(n)
        model.fit(train_data, padded_texts)

        return model

    def make_model(self, e):

        target_triples = set(e.triples)

        texts = []

        for e_ in self.db:
            if target_triples.intersection(e_.triples):
                for l in e_.lexes:
                    texts.append(l['text'])

        tokenized_texts = [normalize_text(x).split() for x in texts]

        n = 3
        train_data, padded_texts = padded_everygram_pipeline(n, tokenized_texts)

        model = Laplace(n)
        model.fit(train_data, padded_texts)

        return model

    def score_text(self, t, model):

        n = 3

        tokenized = normalize_text(t).split()

        return sum(model.logscore(trigram[-1], trigram[:-1])
                   for trigram in ngrams(pad_both_ends(tokenized, n=n), n=n)) / len(tokenized)

    def make_text(self, entry):

        texts = self.make_texts(entry)

        return texts[0]

    def make_fallback_text(self, entry):

        ctx = {'seen': set()}
        sent_texts = [self.fallback_template.fill([a], self.referrer, ctx)
                      for a in entry.triples]
        return ' '.join(sent_texts)

    def get_templates(self, entry, triples):

        a_triples = abstract_triples(triples)
        c_key = (entry.category, a_triples)

        if c_key in self.template_db:
            return self.template_db[c_key]

        templates = list(flatten(self.template_db.get((c, a_triples), [])
                                 for c in self.categories))

        if templates:
            return templates
        elif len(triples) == 1:
            return [self.fallback_template]

        return None

    def make_texts(self, entry):

        texts = []
        n_triples = len(entry.triples)
        n_dp_used = 0

        for dp in self.select_discourse_planning(entry, n_triples):
            if n_dp_used == self.max_dp:
                break

            is_sa_used = False
            n_sa_used = 0
            for sa in self.select_sentence_aggregation(dp, n_triples):
                if n_sa_used == self.max_sa:
                    break

                templates = []

                for sa_part in sa:

                    ts = self.get_templates(entry, sa_part)

                    if ts:
                        sts = self.select_templates(ts, sa_part)
                        templates.append(sts)
                    else:
                        break

                if len(templates) < len(sa):
                    continue

                n_sa_used += 1
                is_sa_used = True

                for ts in product(*templates):

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

                        texts.append(' '.join(sents))

            if is_sa_used:
                n_dp_used += 1

        if texts:
            model = self.make_model(entry)
            if len(model.vocab) == 0:
                model = self.fallback_txs_lm
            texts = sorted(texts,
                           key=partial(
                                   self.score_text,
                                   model=model),
                           reverse=True)
        else:
            texts = [self.make_fallback_text(entry)]

        return texts


def make_model(params, train_set):
    train = load_dataset('train')
    dev = load_dataset('dev')
    db = list(flatten([train, dev]))

    # 1. Grid Search
    # 1.1. Language Models
    # 1.1.1 Template Selection Language Model
    tems_lm = load_template_selection_lm(train_set,
                                         params['tems_lm_n'],
                                         params['tems_lm_name'])
    tems_lm_preprocess_input = load_preprocessing(
            params['tems_lm_preprocess_input'])
    # 1.2 Template database
    template_db = load_template_db(train_set, params.get('tdb_ns', None))
    # 1.3 Referring Expression Generation
    reg = load_referrer(train_set, params['referrer'])
    # 1.4 Discourse Planning
    dp_scorer = load_discourse_planning(train_set, params['dp_scorer'])
    # 1.5 Sentence Aggregation
    sa_scorer = load_sentence_aggregation(train_set, params['sa_scorer'])
    # 1.6 Template Fallback
    fallback_template = load_template_fallback(train_set,
                                               params['fallback_template'])

    # Model
    tgp = TextGenerationPipeline(
            db,
            template_db,
            tems_lm,
            params['tems_lm_bos'],
            params['tems_lm_eos'],
            tems_lm_preprocess_input,
            dp_scorer,
            sa_scorer,
            params['max_dp'],
            params['max_sa'],
            params['max_tems'],
            params['max_refs'],
            fallback_template,
            reg)

    return tgp
