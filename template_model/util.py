# -*- coding: utf-8 -*-
import re
import xml.etree.ElementTree as ET


PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')
CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')


def preprocess_so(so):

    parenthesis_preprocessed = PARENTHESIS_RE.sub(r'\g<2> \g<1>', so)
    underline_removed = parenthesis_preprocessed.replace('_', ' ')
    camelcase_preprocessed = CAMELCASE_RE.sub(r'\g<1> \g<2>',
                                              underline_removed)

    return camelcase_preprocessed.strip('" ')


def extract_triples(entry_elem):

    triples = []

    modifiedtripleset_elem = entry_elem.find('modifiedtripleset')

    for t in modifiedtripleset_elem.findall('mtriple'):

        triple_dict = {}

        for triple_key, part in zip(['subject', 'predicate', 'object'],
                                    t.text.split('|')):

            stripped_part = part.strip()

            triple_dict[triple_key] = stripped_part

        triples.append(triple_dict)

    return triples


def make_test_pkl():

    entries = []

    tree = ET.parse('../evaluation/testdata_with_lex.xml')
    root = tree.getroot()

    for entry_elem in root.iter('entry'):

        entry = {}
        entry['eid'] = entry_elem.attrib['eid']
        entry['category'] = entry_elem.attrib['category']
        entry['triples'] = extract_triples(entry_elem)

        entries.append(entry)

    import pickle

    with open('../evaluation/test.pkl', 'wb') as f:
        pickle.dump(entries, f)


def clear_dir(dirpath):

    import glob
    import os

    files = glob.glob(f'{dirpath}/*')
    for f in files:
        os.remove(f)
