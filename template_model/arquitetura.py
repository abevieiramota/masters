# -*- coding: utf-8 -*-
from itertools import product


class OverPipeline:

    def __init__(self, initial_module, selector):

        self.initial_module = initial_module
        self.selector = selector

    def run(self, i):

        outputs, decisions = self.initial_module.generate([i])

        return self.selector(outputs), decisions


class Module:

    def __init__(self, name, generator, sorter, n_max, next_module=None):

        self.generator = generator
        self.sorter = sorter
        self.n_max = n_max
        self.next_module = next_module
        self.name = name

    def generate(self, flow_chain):

        decisions = []

        sorted_outputs = self.sorter(self.generator(flow_chain), flow_chain)

        if not self.next_module:

            return sorted_outputs[:self.n_max], (self.name, sorted_outputs[:self.n_max])

        results = []

        for curr_o in sorted_outputs[:self.n_max]:

            curr_flow_chain = flow_chain + [curr_o]

            result, next_decisions = self.next_module.generate(curr_flow_chain)

            decisions.append((self.name, curr_o, next_decisions))

            if result:
                results.extend(result)

        return results, decisions


class MultiModule:

    def __init__(self, name, generator, sorter, n_max, next_module=None):

        self.generator = generator
        self.sorter = sorter
        self.n_max = n_max
        self.next_module = next_module
        self.name = name

    def generate(self, flow_chain):

        i = flow_chain[-1]

        all_partial_results = []

        decisions = []

        for i_part in i:

            curr_flow_chain = flow_chain[:-1] + [i_part]

            part_sorted_outputs = self.sorter(self.generator(curr_flow_chain),
                                              curr_flow_chain)

            all_partial_results.append(part_sorted_outputs[:self.n_max])

        results = []

        for curr_o in product(*all_partial_results):

            curr_flow_chain = flow_chain + [curr_o]

            result, next_decisions = self.next_module.generate(curr_flow_chain)

            decisions.append((self.name, curr_o, next_decisions))

            if result:
                results.extend(result)

        return results, decisions
