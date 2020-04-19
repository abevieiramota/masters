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
        # FIX: a entrada dev/3triples/SportsTeam Id4 está sem textos de referência
        #   isso na base do Thiago, na base original ela tem as referências
        if not t:
            continue
        tem = l['template']

        m = lexicalization_match(t, tem, e.entity_map)

        if not m:
            w_error.append(l)
        else:

            map_key_to_lex = {k.rsplit('_', 1)[0].replace('_', '-'): v
                              for k, v in m.groupdict().items()}

            map_lex_to_key = {v: k for k, v in map_key_to_lex.items()}

            ref_regions = m.regs[1:]

            all_regions = []
            if ref_regions[0][0] != 0:
                ini = 0
                end = ref_regions[0][0]
                text = t[ini: end]
                all_regions.append((ini, 'text', text))
            for r1, r2 in zip(ref_regions[:-1], ref_regions[1:]):
                ini = r1[1]
                end = r2[0]
                text = t[ini: end]
                all_regions.append((ini, 'text', text))
            for r, key in zip(ref_regions, map_lex_to_key.values()):
                ini = r[0]
                end = r[1]
                entity_id = e.entity_map[key]
                captured = t[ini: end]
                all_regions.append((ini, 'ref', entity_id, captured))
            if ref_regions[-1][1] != len(t):
                ini = ref_regions[-1][1]
                end = len(t)
                text = t[ini: end]
                all_regions.append((ini, 'text', text))
            all_regions = sorted(all_regions)

            for i in range(len(all_regions)):

                current_reg = all_regions[i]

                if current_reg[1] == 'ref':

                    captured = current_reg[3]
                    key = current_reg[2]
                    right_regions = all_regions[:i]
                    right_context = ''.join(r[2] for r in right_regions)
                    left_regions = all_regions[i+1:]
                    left_context = ''.join(l[2] for l in left_regions)

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
