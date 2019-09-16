# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions
from random import Random
import pickle
from template_based import JustJoinTemplate, abstract_triples
from reg import REGer, load_name_db, load_pronoun_db
from arquitetura import Module, OverPipeline, MultiModule
from experimento_discourse_planning import get_dp_sorter
from experimento_sentence_aggregation import get_sa_sorter
from util import Entry
import sys
import os
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu

# montar o pipeline


def make_pipe(use_lm):

    if use_lm:

        import kenlm

        lm = kenlm.Model('../data/kenlm/lm.arpa').score
        ts_lm = kenlm.Model('../data/kenlm/ts_lm.arpa').score
    else:
        lm = lambda x: Random().choice(x)
        ts_lm = lambda x: Random().choice(x)

    def random_sorter(x, flow_chain, random_seed=None):

        x_copy = x[::]

        Random(random_seed).shuffle(x_copy)

        return x_copy

    pronoun_db = load_pronoun_db()
    name_db = load_name_db()
    refer = REGer(pronoun_db, name_db).refer

    def tg_gen(flow_chain):

        agg, templates = flow_chain[-2:]

        ctx = {'seen': set()}

        texts = [t.fill(a, refer, ctx)
                 for a, t in zip(agg, templates)]

        return [' '.join(texts)]

    tg_sorter = lambda x, flow_chain: x
    tg_n_max = 2
    tg = Module(tg_gen, tg_sorter, tg_n_max, None)

    with open('../data/templates/template_db/tdb', 'rb') as f:
        template_db = pickle.load(f)

    def ts_gen(flow_chain):

        agg_part = flow_chain[-1]
        e = flow_chain[0]

        abstracted_part = abstract_triples(agg_part)
        template_key = (e.category, abstracted_part)

        if template_key in template_db:
            templates = template_db[template_key]

            return list(templates)
        elif len(abstracted_part) == 1:

            return [JustJoinTemplate()]
        else:
            return []

    def ts_sorter(ts, flow_chain):

        agg_part = flow_chain[-1]

        ts_result = sorted(ts,
                           key=lambda t: ts_lm(t.fill(agg_part,
                                                      lambda so, ctx: so,
                                                      None)),
                           reverse=True)

        return ts_result

    ts_n_max = 3
    ts = MultiModule(ts_gen, ts_sorter, ts_n_max, tg)

    sa_gen = lambda flow_chain: list(partitions(flow_chain[-1]))
    sa_sorter = get_sa_sorter()
    sa_n_max = 1
    sa = Module(sa_gen, sa_sorter, sa_n_max, ts)

    dp_gen = lambda flow_chain: list(permutations(flow_chain[-1].triples))
    dp_sorter = get_dp_sorter()
    dp_n_max = 1
    dp = Module(dp_gen, dp_sorter, dp_n_max, sa)

    pipe_selector = lambda x: max(x, key=lm)

    pipe = OverPipeline(dp, pipe_selector)

    return pipe

# gerar os textos e salvar em arquivo


def make_texts(pipe, entries, outpath):

    with open(outpath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries):
            text = pipe.run(e)
            f.write(f'{text}\n')
            if i % 10 == 0:
                print(i)


def do_all(entries, model_name):

    outdir = f'../data/models/{model_name}'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    outpath = f'../data/models/{model_name}/{model_name}.txt'

    pipe = make_pipe(True)
    make_texts(pipe, entries, outpath)

    preprocess_model_to_evaluate(outpath)

    return bleu(model_name, 'old-cat', [0, 1, 2])


if __name__ == '__main__':

    from util import load_shared_task_test

    test = load_shared_task_test()

    pipe = make_pipe(False)
    pipe.run(test[941])

    #score = do_all(test, 'abe-random')

    #print(score)
