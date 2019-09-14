# -*- coding: utf-8 -*-


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

    def generate(self, flow_chain):

        n_triples = len(flow_chain[0].triples)

        sorted_outputs = self.sorter(self.generator(flow_chain), n_triples)

        if not self.next_module:

            return sorted_outputs[:self.n_max]

        total_consumed = 0
        total_tries = 0
        results = []

        while total_consumed < self.n_max and sorted_outputs:

            curr_o = sorted_outputs.pop(0)

            curr_flow_chain = flow_chain + [curr_o]

            result = self.next_module.generate(curr_flow_chain)

            total_tries += 1

            if result:
                total_consumed += 1
                results.extend(result)

        self.n_tries.append(total_tries)

        return results
