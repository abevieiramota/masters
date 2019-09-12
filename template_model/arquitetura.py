# -*- coding: utf-8 -*-
from itertools import permutations, product
from more_itertools import partitions, collapse
from random import Random
import pickle
from template_based import JustJoinTemplate, abstract_triples
from reg import REGer, load_name_db, load_pronoun_db


class Pipeline:

    def __init__(self, initial_module, selector):

        self.initial_module = initial_module
        self.selector = selector

    def run(self, i):

        outputs = self.initial_module.generate([i])

        return self.selector(outputs)


class Module:

    def __init__(self, generator, sorter, n_max, next_module=None):

        self.generator = generator
        self.sorter = sorter
        self.n_max = n_max
        self.next_module = next_module

    def generate(self, flow_chain):

        sorted_outputs = self.sorter(self.generator(flow_chain))

        if not self.next_module:

            return sorted_outputs[:self.n_max]

        total_consumed = 0
        results = []

        while total_consumed < self.n_max and sorted_outputs:

            curr_o = sorted_outputs.pop(0)

            curr_flow_chain = flow_chain + [curr_o]

            result = self.next_module.generate(curr_flow_chain)

            if result:
                total_consumed += 1
                results.extend(result)

        return results


def random_sorter(x, random_seed=None):

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


tg_sorter = random_sorter
tg_n_max = 1
tg = Module(tg_gen, tg_sorter, tg_n_max, None)


with open('../data/templates/template_db/tdb', 'rb') as f:
    template_db = pickle.load(f)


def ts_gen(flow_chain):

    perm, agg = flow_chain[-2:]
    e = flow_chain[0]

    parts_templates = []

    for agg_part in agg:

        abstracted_part = abstract_triples(agg_part)
        template_key = (e.category, abstracted_part)

        if template_key in template_db:
            templates = template_db[template_key]
            parts_templates.append(templates)

        elif len(abstracted_part) == 1:
            parts_templates.append([JustJoinTemplate()])
        else:
            return []

    return list(product(*parts_templates))


ts_sorter = random_sorter
ts_n_max = 2
ts = Module(ts_gen, ts_sorter, ts_n_max, tg)

sa_gen = lambda flow_chain: list(partitions(flow_chain[-1]))
sa_sorter = random_sorter
sa_n_max = 2
sa = Module(sa_gen, sa_sorter, sa_n_max, ts)

dp_gen = lambda flow_chain: list(permutations(flow_chain[-1].triples))
dp_sorter = random_sorter
dp_n_max = 2
dp = Module(dp_gen, dp_sorter, dp_n_max, sa)

pipe = Pipeline(dp, lambda x: x[0])