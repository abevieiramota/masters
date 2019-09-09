# -*- coding: utf-8 -*-
import re
import xml.etree.ElementTree as ET
from reading_thiagos_templates import (
        extract_triples,
        extract_lexes,
        extract_entity_map)
from collections import namedtuple
import glob
import pickle
import os


V_15_BASEPATH = '../../webnlg/data/v1.5/en/'


PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')
CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')


Entry = namedtuple('Entry', ['eid',
                             'category',
                             'triples',
                             'lexes',
                             'entity_map',
                             'r_entity_map'])


def preprocess_so(so):

    parenthesis_preprocessed = PARENTHESIS_RE.sub(r'\g<2> \g<1>', so)
    underline_removed = parenthesis_preprocessed.replace('_', ' ')
    camelcase_preprocessed = CAMELCASE_RE.sub(r'\g<1> \g<2>',
                                              underline_removed)

    return camelcase_preprocessed.strip('" ')


def make_test_pkl():

    filepaths = glob.glob(os.path.join(V_15_BASEPATH, 'test/**/*.xml'),
                          recursive=True)

    entries = []

    for fp in filepaths:

        tree = ET.parse(fp)
        root = tree.getroot()

        for entry_elem in root.iter('entry'):

            eid = entry_elem.attrib['eid']
            category = entry_elem.attrib['category']
            triples = extract_triples(entry_elem)
            lexes = extract_lexes(entry_elem)
            entity_map = extract_entity_map(entry_elem)
            r_entity_map = {v: k for k, v in entity_map.items()}

            entries.append(Entry(eid,
                                 category,
                                 triples,
                                 lexes,
                                 entity_map,
                                 r_entity_map))

    with open('../evaluation/test.pkl', 'wb') as f:
        pickle.dump(entries, f)


def make_shared_task_test_pkl():

    entries = []

    tree = ET.parse('../evaluation/testdata_with_lex.xml')
    root = tree.getroot()

    for entry_elem in root.iter('entry'):

        eid = entry_elem.attrib['eid']
        category = entry_elem.attrib['category']
        triples = extract_triples(entry_elem)

        entries.append(Entry(eid, category, triples, None, None, None))

    with open('../evaluation/test_shared_task.pkl', 'wb') as f:
        pickle.dump(entries, f)


def make_train_dev_pkl():

    filepaths = glob.glob(os.path.join(V_15_BASEPATH, 'train/**/*.xml'),
                          recursive=True)
    filepaths.extend(glob.glob(os.path.join(V_15_BASEPATH, 'dev/**/*.xml'),
                               recursive=True))

    entries = []

    for fp in filepaths:

        tree = ET.parse(fp)
        root = tree.getroot()

        for entry_elem in root.iter('entry'):

            eid = entry_elem.attrib['eid']
            category = entry_elem.attrib['category']
            triples = extract_triples(entry_elem)
            lexes = extract_lexes(entry_elem)
            entity_map = extract_entity_map(entry_elem)
            r_entity_map = {v: k for k, v in entity_map.items()}

            entries.append(Entry(eid,
                                 category,
                                 triples,
                                 lexes,
                                 entity_map,
                                 r_entity_map))

    with open('../evaluation/train_dev.pkl', 'wb') as f:
        pickle.dump(entries, f)


def make_train_pkl():

    filepaths = glob.glob(os.path.join(V_15_BASEPATH, 'train/**/*.xml'),
                          recursive=True)

    entries = []

    for fp in filepaths:

        tree = ET.parse(fp)
        root = tree.getroot()

        for entry_elem in root.iter('entry'):

            eid = entry_elem.attrib['eid']
            category = entry_elem.attrib['category']
            triples = extract_triples(entry_elem)
            lexes = extract_lexes(entry_elem)
            entity_map = extract_entity_map(entry_elem)
            r_entity_map = {v: k for k, v in entity_map.items()}

            entries.append(Entry(eid,
                                 category,
                                 triples,
                                 lexes,
                                 entity_map,
                                 r_entity_map))

    with open('../evaluation/train.pkl', 'wb') as f:
        pickle.dump(entries, f)


def clear_dir(dirpath):

    import glob
    import os

    files = glob.glob(f'{dirpath}/*')
    for f in files:
        os.remove(f)


def load_test():

    with open('../evaluation/test.pkl', 'rb') as f:
        test = pickle.load(f)

    return test


def load_shared_task_test():

    with open('../evaluation/test_shared_task.pkl', 'rb') as f:
        test = pickle.load(f)

    return test


def load_train_dev():

    with open('../evaluation/train_dev.pkl', 'rb') as f:
        td = pickle.load(f)

    return td
