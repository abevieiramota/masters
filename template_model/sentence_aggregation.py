# -*- coding: utf-8 -*-
from more_itertools import partitions


class PartitionsSentenceAggregation:

    def agg(self, entry, seqs):

        all_partitions = sorted(partitions(seqs),
                                key=lambda e: [len(e)] +
                                [i*len(z)*-1 for i, z in enumerate(e)])

        for partition in all_partitions:

            yield partition
