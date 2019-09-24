# -*- coding: utf-8 -*-
from reading_thiagos_templates import extract_refs, load_dataset, Entry
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(BASE_DIR, '../data/pretrained_models')
REFERRER_DB_FILENAME = 'referrer_pretrained_counter_{}'


def load_name_pronoun_db(dataset_name, name):

    if name == 'referrer_pretrained':
        referrer_db_filepath = os.path.join(PRETRAINED_DIR,
                                            REFERRER_DB_FILENAME.format(
                                                    dataset_name))
        with open(referrer_db_filepath, 'rb') as f:
            data = pickle.load(f)

        return data['name_db'], data['pronoun_db']


def make_pretrained_name_pronoun_db(dataset_name):

    dataset = load_dataset(dataset_name)

    referrer_db_filepath = os.path.join(PRETRAINED_DIR,
                                        REFERRER_DB_FILENAME.format(
                                                dataset_name))

    name_db, pronoun_db = extract_refs(dataset)

    with open(referrer_db_filepath, 'wb') as f:
        data = {'name_db': name_db,
                'pronoun_db': pronoun_db}
        pickle.dump(data, f)
