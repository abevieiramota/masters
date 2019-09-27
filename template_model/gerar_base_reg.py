# -*- coding: utf-8 -*-
from reading_thiagos_templates import (
        load_dataset,
        lexicalization_match
)
from more_itertools import flatten
import csv
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')


def extract_data(e):

    data = []
    w_error = []

    for l in e.lexes:
        t = l['text']
        tem = l['template']

        m = lexicalization_match(t, tem, e.entity_map)

        if not m:
            w_error.append(l)
        else:

            map_lex_to_key = {v: k.rsplit('_', 1)[0].replace('_', '-')
                              for k, v in m.groupdict().items()}

            for region in m.regs[1:]:

                captured = t[region[0]: region[1]]
                var_name = map_lex_to_key[captured]
                key = e.entity_map[var_name]
                right_context = t[:region[0]]
                left_context = t[region[1]:]

                data.append({
                        'wiki': key,
                        'ref_expression': captured,
                        'right_context': right_context,
                        'left_context': left_context
                        })

    return data, w_error


# TODO: Possui erros!!!!!
def make_datasets():

    for dataset_name in ['train', 'test', 'dev']:

        dataset = load_dataset(dataset_name)

        data, errors = zip(*(extract_data(e) for e in dataset))
        data = flatten(data)
        errors = flatten(errors)

        dataset_filename = 'reg_{}.csv'.format(dataset_name)
        dataset_filepath = os.path.join(PRETRAINED_DIR,
                                        dataset_filename)

        fieldnames = ['wiki',
                      'ref_expression',
                      'right_context',
                      'left_context']
        with open(dataset_filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=fieldnames,
                                    delimiter='\t')
            writer.writeheader()
            writer.writerows(data)

    return errors
