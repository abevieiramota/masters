# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, sort_together
from template_based import JustJoinTemplate, abstract_triples
#from experimento_discourse_planning import get_dp_scorer
#from experimento_sentence_aggregation import get_sa_scorer
from functools import partial
import sys
from random import shuffle
import os
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu


def get_random_scores(n):

    rs = list(range(n))
    shuffle(rs)

    return rs


def random_dp_scorer(dps, n_triples):

    return get_random_scores(len(dps))


def random_sa_scorer(sas, n_triples):

    rs = get_random_scores(len(sas))

    ix_1_triple_1_sen = [i for i, sa in enumerate(sas)
                         if len(sa) == n_triples]

    rs[ix_1_triple_1_sen[0]] = 10e5

    return rs


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

        text = t.fill(a, lambda so, ctx: so, None)
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

                    texts.append(' '.join(sent_texts))

        texts = sorted(texts,
                       key=self.score_text,
                       reverse=True)

        return texts



def make_texts(entries, model, outpath):

    with open(outpath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries):
            text = model(e)[0]
            f.write(f'{text}\n')
            if i % 10 == 0:
                print(i)


def do_all(set_, model_name):

    data = load_dataset(set_)

    outdir = f'../data/models/{set_}/{model_name}'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outpath = f'../data/models/{set_}/{model_name}/{model_name}.txt'

    model = load_model(set_)

    make_texts(data, model, outpath)

    preprocess_model_to_evaluate(outpath, set_)

    return bleu(model_name, set_, 'old-cat', [0, 1, 2])


if __name__ == '__main__':

    set_ = 'dev'

    score = do_all(set_, 'abe-random2')

    print(score)
