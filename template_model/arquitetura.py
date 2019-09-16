# -*- coding: utf-8 -*-
from itertools import product


class OverPipeline:

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
        self.n_tries = []
        self.consumed = None

    def generate(self, flow_chain):

        sorted_outputs = self.sorter(self.generator(flow_chain), flow_chain)

        if not self.next_module:

            return sorted_outputs[:self.n_max]

        self.consumed = []
        total_tries = 0
        results = []

        while len(self.consumed) < self.n_max and sorted_outputs:

            curr_o = sorted_outputs.pop(0)

            curr_flow_chain = flow_chain + [curr_o]

            result = self.next_module.generate(curr_flow_chain)

            total_tries += 1

            if result:
                #self.consumed.append(curr_o)
                results.extend(result)

        self.n_tries.append(total_tries)

        return results


class MultiModule:

    def __init__(self, generator, sorter, n_max, next_module=None):

        self.generator = generator
        self.sorter = sorter
        self.n_max = n_max
        self.next_module = next_module
        self.n_tries = []
        self.consumed = None

    def generate(self, flow_chain):

        self.consumed = []

        i = flow_chain[-1]

        all_partial_results = []

        for i_part in i:

            curr_flow_chain = flow_chain[:-1] + [i_part]

            part_sorted_outputs = self.sorter(self.generator(curr_flow_chain),
                                              curr_flow_chain)

            all_partial_results.append(part_sorted_outputs[:self.n_max])

        results = []

        for curr_o in product(*all_partial_results):

            curr_flow_chain = flow_chain + [curr_o]

            result = self.next_module.generate(curr_flow_chain)

            if result:
                #self.consumed.append(curr_o)
                results.extend(result)

        return results
