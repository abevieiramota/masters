# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, flatten
from random import Random
import pickle
from template_based import JustJoinTemplate, abstract_triples
from reg import REGer, load_name_db, load_pronoun_db
from arquitetura import Module, OverPipeline, MultiModule
from experimento_discourse_planning import get_dp_scorer
from experimento_sentence_aggregation import get_sa_scorer
from util import Entry, preprocess_so
from functools import partial
import sys
import os
sys.path.append('../evaluation')
from evaluate import preprocess_model_to_evaluate, bleu

# montar o pipeline


def make_pipe(use_lm):

    def random_scorer(x, flow_chain, random_seed=None):

        ixs = list(range(len(x)))
        Random(random_seed).shuffle(ixs)

        return ixs

    if use_lm:

        import kenlm

        lm_1 = partial(kenlm.Model('../data/kenlm/lm_1.arpa').score,
                       bos=False,
                       eos=False)
        lm_gt1 = partial(kenlm.Model('../data/kenlm/lm_gt1.arpa').score,
                         bos=False,
                         eos=False)
        ts_1_lm = partial(kenlm.Model('../data/kenlm/ts_1_lm.arpa').score,
                          bos=False,
                          eos=False)
        ts_gt1_lm = partial(kenlm.Model('../data/kenlm/ts_gt1_lm.arpa').score,
                            bos=False,
                            eos=False)
    else:
        lm_1 = lambda x: Random().randint(0, 1000)
        lm_gt1 = lambda x: Random().randint(0, 1000)
        ts_1_lm = lambda x: Random().randint(0, 1000)
        ts_gt1_lm = lambda x: Random().randint(0, 1000)

    #pronoun_db = load_pronoun_db()
    name_db = load_name_db()
    #refer = REGer(pronoun_db, name_db).refer

    def tg_gen(flow_chain):

        templates, regs = flow_chain[-2:]

        texts = [t.fill_(r)
                 for t, r in zip(templates, regs)]

        return [' '.join(texts)]

    tg_scorer = lambda x, flow_chain: [-1]
    tg_n_max = 1
    tg = Module('TG', tg_gen, tg_scorer, tg_n_max, None)

    def reg_gen(flow_chain):

        agg_part, template = flow_chain[-2:]

        aligned_data = template.align(agg_part[0])

        reg_data = {}
        for slot, so in aligned_data.items():
            if so in name_db:
                reg_so = name_db[so].most_common()[0][0]
            else:
                reg_so = preprocess_so(so)

            reg_data[slot] = reg_so

        return [reg_data]

    def reg_scorer(x, flow_chain):

        return [Random().randint(0, 1000) for _ in range(len(x))]

    reg_n_max = 2
    reg = MultiModule('REG', reg_gen, reg_scorer, reg_n_max, tg)

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

    def ts_scorer(ts, flow_chain):

        agg_part = flow_chain[-1]

        ts_result = []

        lm = ts_gt1_lm if len(agg_part) == 1 else ts_1_lm

        ts_result = [lm(t.fill(agg_part, lambda so, ctx: so, None).lower())
                     for t in ts]

        return ts_result

    ts_n_max = 5
    ts = MultiModule('TS', ts_gen, ts_scorer, ts_n_max, reg)

    sa_gen = lambda flow_chain: list(partitions(flow_chain[-1]))
    sa_scorer = get_sa_scorer()
    sa_n_max = 3
    sa = Module('SA', sa_gen, sa_scorer, sa_n_max, ts)

    dp_gen = lambda flow_chain: list(permutations(flow_chain[-1].triples))
    dp_scorer= get_dp_scorer()
    dp_n_max = 3
    dp = Module('DP', dp_gen, dp_scorer, dp_n_max, sa)

    def pipe_scorer(texts, e):

        len_e = len(e.triples)

        if len_e == 1:
            return [lm_gt1(x.lower()) for x in texts]
        else:
            return [lm_1(x.lower()) for x in texts]

    pipe = OverPipeline(dp, pipe_scorer)

    return pipe

# gerar os textos e salvar em arquivo


def make_texts(pipe, entries, outpath):

    with open(outpath, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries):
            text, _ = pipe.run(e)
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
    pipe.run(test[690])

    #score = do_all(test, 'abe-random')

    #print(score)
