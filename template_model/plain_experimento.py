# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, sort_together
import pickle
from template_based import JustJoinTemplate, abstract_triples
from reg import REGer, load_name_db, load_pronoun_db
from experimento_discourse_planning import get_dp_scorer
from experimento_sentence_aggregation import get_sa_scorer
from functools import partial
from util import Entry, preprocess_so
import sys
import kenlm
from random import shuffle
import os
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu


def get_random_scores(n):

    rs = list(range(n))
    shuffle(rs)

    return rs

dp_scorer = get_dp_scorer()
#dp_scorer = lambda dps, n_triples: get_random_scores(len(dps))
MAX_DP = 2

def random_sa_scorer(sas, n_triples):

    rs = get_random_scores(len(sas))

    ix_1_triple_1_sen = [i for i, sa in enumerate(sas) if len(sa) == n_triples]

    rs[ix_1_triple_1_sen[0]] = 10e5

    return rs

sa_scorer = get_sa_scorer()
#sa_scorer = random_sa_scorer
MAX_SA = 3

with open('../data/templates/template_db/tdb', 'rb') as f:
    template_db = pickle.load(f)
MAX_TS = 5
jjt = JustJoinTemplate()
ts_lm = partial(kenlm.Model('../data/kenlm/ts_lm.arpa').score,
                bos=False,
                eos=False)

pronoun_db = load_pronoun_db()
name_db = load_name_db()
refer = REGer(pronoun_db, name_db).refer
#refer = lambda so, ctx: preprocess_so(so)

lm = partial(kenlm.Model('../data/kenlm/refs_lm.arpa').score,
             bos=False,
             eos=False)


def score_template(t, a):

    text = t.fill(a, lambda so, ctx: so, None)
    return ts_lm(text.lower())


def make_text(entry):

    n_triples = len(entry.triples)

    texts = []

    dps = list(permutations(entry.triples))
    dps_scores = dp_scorer(dps, n_triples)
    dps = sort_together([dps_scores, dps],
                        reverse=True)[1]

    for dp in dps[:MAX_DP]:

        sas = list(partitions(dp))
        sas_scores = sa_scorer(sas, n_triples)
        sas = sort_together([sas_scores, sas],
                            reverse=True)[1]

        for sa in sas[:MAX_SA]:

            templates = []

            for sa_part in sa:

                a_part = abstract_triples(sa_part)
                t_key = (entry.category, a_part)

                if t_key in template_db:
                    t_key_ts = template_db[t_key]

                    t_key_ts = sorted(t_key_ts,
                                      key=lambda t: score_template(t, sa_part),
                                      reverse=True)

                    templates.append(t_key_ts[:MAX_TS])
                elif len(a_part) == 1:
                    templates.append([jjt])
                else:
                    templates.append([])

            for ts in product(*templates):

                ctx = {'seen': set()}

                sent_texts = [t.fill(a, refer, ctx)
                              for a, t in zip(sa, ts)]

                texts.append(' '.join(sent_texts))

    texts = sorted(texts,
                   key=lambda t: lm(t.lower()),
                   reverse=True)

    return texts


def make_texts(entries, outpath):

    with open(outpath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries):
            text = make_text(e)[0]
            f.write(f'{text}\n')
            if i % 10 == 0:
                print(i)


def do_all(entries, model_name):

    outdir = f'../data/models/{model_name}'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outpath = f'../data/models/{model_name}/{model_name}.txt'

    make_texts(entries, outpath)

    preprocess_model_to_evaluate(outpath)

    return bleu(model_name, 'old-cat', [0, 1, 2])


if __name__ == '__main__':

    from util import load_shared_task_test

    test = load_shared_task_test()

    score = do_all(test, 'abe-random2')

    print(score)
