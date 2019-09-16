# -*- coding: utf-8 -*-
from itertools import product


class OverPipeline:

    def __init__(self, initial_module, selector):

        self.initial_module = initial_module
        self.selector = selector

    def run(self, i):

        outputs, decisions = self.initial_module.generate([i])

        return self.selector(outputs), decisions


# TODO: use a more efficient data structure, like a stack
class Module:

    def __init__(self, name, generator, scorer, n_max, next_module=None):

        self.generator = generator
        self.scorer = scorer
        self.n_max = n_max
        self.next_module = next_module
        self.name = name

    def generate(self, flow_chain):

        decisions = []

        possible_cases = self.generator(flow_chain)

        scores = self.scorer(possible_cases, flow_chain)

        sorted_outputs = sorted(zip(possible_cases, scores),
                                key=lambda x: x[1],
                                reverse=True)

        if not self.next_module:

            results = [x[0] for x in sorted_outputs[:self.n_max]]
            decisions = [(self.name, x[0], x[1], None)
                         for x in sorted_outputs[:self.n_max]]

            return results, decisions

        results = []

        for curr_o, score in sorted_outputs[:self.n_max]:

            curr_flow_chain = flow_chain + [curr_o]

            result, next_decisions = self.next_module.generate(curr_flow_chain)

            decisions.append((self.name, curr_o, score, next_decisions))

            if result:
                results.extend(result)

        return results, decisions


class MultiModule:

    def __init__(self, name, generator, scorer, n_max, next_module=None):

        self.generator = generator
        self.scorer = scorer
        self.n_max = n_max
        self.next_module = next_module
        self.name = name

    def generate(self, flow_chain):

        i = flow_chain[-1]

        all_partial_results = []

        decisions = []

        for i_part in i:

            curr_flow_chain = flow_chain[:-1] + [i_part]

            possible_cases = self.generator(curr_flow_chain)

            scores = self.scorer(possible_cases, curr_flow_chain)

            part_sorted_outputs = sorted(zip(possible_cases, scores),
                                         key=lambda x: x[1],
                                         reverse=True)

            all_partial_results.append(part_sorted_outputs[:self.n_max])

        results = []

        for curr_o_scores in product(*all_partial_results):

            curr_o = [x[0] for x in curr_o_scores]
            scores = [x[1] for x in curr_o_scores]

            curr_flow_chain = flow_chain + [curr_o]

            result, next_decisions = self.next_module.generate(curr_flow_chain)

            decisions.append((self.name, curr_o, scores, next_decisions))

            if result:
                results.extend(result)

        return results, decisions
