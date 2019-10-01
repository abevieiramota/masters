import re
from template_based import Template, Triple, abstract_triples
from collections import defaultdict, Counter, namedtuple
from nltk import sent_tokenize
import pickle
import glob
import xml.etree.ElementTree as ET
import os


# usado na extração de expressões de referência
#    alinhando texto e template
#    tem \\ porque é aplicado sobre o template após um re.escape
RE_MATCH_TEMPLATE_KEYS_LEX = re.compile((r'(AGENT\\-\d|PATIENT\\-\d'
                                         r'|BRIDGE\\-\d)'))
RE_MATCH_TEMPLATE_KEYS = re.compile(r'(AGENT\-\d|PATIENT\-\d|BRIDGE\-\d)')
TRANS_ESCAPE_TO_RE = str.maketrans('-', '_', '\\')

RE_SPACE_BEFORE_COMMA_DOT = re.compile(r'(?<=\b)\s(?=\.|,)')

RE_WEIRD_QUOTE_MARKS = re.compile(r'(`{1,2})|(\'{1,2})')

# {{}} -> é só para escapar as chaves
SLOT_PLACEHOLDER = '{{{}-{}-{}}}'

V_15_BASEPATH = '../../webnlg/data/v1.5/en/'


Entry = namedtuple('Entry', ['eid',
                             'category',
                             'triples',
                             'lexes',
                             'entity_map'])

MAP_REF_TYPE_TO_KEY = dict(description='D',
                           name='N',
                           pronoun='P',
                           demonstrative='E')


def extract_templates(dataset):

    entries_templates = []
    extraction_error = []

    dataset_w_entity_map = [e for e in dataset if e.entity_map]

    for e in dataset_w_entity_map:

        good_lexes = [l
                      for l in e.lexes
                      if (
                              l['comment'] == 'good'
                              and l['template']
                              and l['sorted_triples']
                              and l['references']
                              )
                      ]

        lexes_templates = []

        for l in good_lexes:

            ts = make_template(l['sorted_triples'],
                               l['template'],
                               e.entity_map,
                               l['references'])

            if not ts:
                extraction_error.append((e, l))
            else:
                lexes_templates.append((l, ts))

        if lexes_templates:
            entries_templates.append((e, lexes_templates))

    return entries_templates, extraction_error


def make_template_lm_texts(entries_templates):

    template_lm_texts = []

    for e, lexes_templates in entries_templates:

        for l, ts in lexes_templates:

            for t, triples in zip(ts, l['sorted_triples']):

                text = t.fill(triples, lambda x, ctx: x, None)

                template_lm_texts.append(text)

    return template_lm_texts


def extract_refs(dataset):

    refs = defaultdict(lambda: defaultdict(lambda: Counter()))

    for e in dataset:

        for l in e.lexes:

            if l['references']:

                for r in l['references']:

                    type_key = MAP_REF_TYPE_TO_KEY[r['type']]

                    refs[type_key][r['entity']][r['ref']] += 1

    return {k: dict(v) for k, v in refs.items()}


def normalize_thiagos_template(s):
    # removes single space between an entity and a dot or a comma

    s = RE_SPACE_BEFORE_COMMA_DOT.sub('', s)

    return RE_WEIRD_QUOTE_MARKS.sub('"', s)


def delexicalize_triples(triples, entity_map):

    delexicalized_triples = []
    s_key_to_entity, o_key_to_entity = {}, {}
    for key, entity in entity_map.items():
        if 'AGENT' in key:
            s_key_to_entity[entity] = key
        if 'PATIENT' in key:
            o_key_to_entity[entity] = key
        if 'BRIDGE' in key:
            s_key_to_entity[entity] = key
            o_key_to_entity[entity] = key

    for t in triples:

        delexicalized_t = Triple(s_key_to_entity[t.subject],
                                 t.predicate,
                                 o_key_to_entity[t.object])

        delexicalized_triples.append(delexicalized_t)

    return delexicalized_triples


def make_template(sorted_triples,
                  template_text,
                  entity_map,
                  references):

    templates = []

    template_sens = sent_tokenize(template_text)

    types_by_key = defaultdict(list)
    keys_in_references = set()

    for ref in references:

        ref_type = MAP_REF_TYPE_TO_KEY[ref['type']]

        types_by_key[ref['tag']].append(ref_type)
        keys_in_references.add(ref['tag'])

    for template_sen, triples_sen in zip(template_sens, sorted_triples):

        delexicalized_triples = delexicalize_triples(triples_sen, entity_map)
        abstracted_triples = abstract_triples(triples_sen)

        map_template_key_to_slot = {}
        for d, a in zip(delexicalized_triples, abstracted_triples):
            map_template_key_to_slot[d.subject] = a.subject
            map_template_key_to_slot[d.object] = a.object

        entity_ref_counter = Counter()

        keys_in_template_sen = RE_MATCH_TEMPLATE_KEYS.findall(template_sen)

        # FIXME: estou ocultando erros
        if set(keys_in_template_sen) - map_template_key_to_slot.keys():
            continue

        for key in keys_in_template_sen:
            i_occurrence = entity_ref_counter[key]
            slot = map_template_key_to_slot[key]
            # FIXME: estou ocultando erros
            if i_occurrence >= len(types_by_key[key]):
                return []
            type_ = types_by_key[key][i_occurrence]
            s_slot = SLOT_PLACEHOLDER.format(slot, i_occurrence, type_)
            template_sen = template_sen.replace(key, s_slot, 1)

            entity_ref_counter[key] += 1

        # FIXME: não sei mais se essa checagem é necessária ou qual a importância dela
        slots = {t.subject
                 for t in abstracted_triples} | {t.object
                                                 for t in abstracted_triples}
        all_slots_in_sen = all(slot in template_sen for slot in slots)

        if all_slots_in_sen:
            # removes @ -> looks like an error
            template_sen = template_sen.replace('@', '')
            t = Template(abstracted_triples, template_sen)

            templates.append(t)
        else:
            return []

    return templates


def extract_triples(entry_elem):

    triples = []

    modifiedtripleset_elem = entry_elem.find('modifiedtripleset')

    for t in modifiedtripleset_elem.findall('mtriple'):

        sub, pred, obj = [x.strip(' ') for x in t.text.split('|')]

        triple = Triple(sub, pred, obj)

        triples.append(triple)

    return tuple(triples)


def extract_entity_map(entry_elem):

    entity_dict = {}

    for ent_elem in entry_elem.find('entitymap').findall('entity'):

        ent_placeholder, ent_value = ent_elem.text.split('|')

        entity_dict[ent_placeholder.strip()] = ent_value.strip(' ')

    return entity_dict


def extract_lexes(entry_elem):

    lexes = []

    for lex_elem in entry_elem.findall('lex'):

        sorted_sent_triples = None
        sorted_tripleset_elem = lex_elem.find('sortedtripleset')

        if sorted_tripleset_elem:

            sorted_sent_triples = []

            for sorted_sent_elem in sorted_tripleset_elem.findall('sentence'):

                sorted_triples = []

                for t in sorted_sent_elem.findall('striple'):
                    sub, pred, obj = [x.strip(' ') for x in t.text.split('|')]

                    triple = Triple(sub, pred, obj)

                    sorted_triples.append(triple)

                if sorted_triples:
                    sorted_sent_triples.append(tuple(sorted_triples))

        refs_elem = lex_elem.find('references')

        references = None
        if refs_elem:

            references = []

            for ref_elem in refs_elem.findall('reference'):

                reference = {
                        'entity': ref_elem.attrib['entity'],
                        'tag': ref_elem.attrib['tag'],
                        'type': ref_elem.attrib['type'],
                        'ref': ref_elem.text}

                references.append(reference)

        lex = {'text': lex_elem.findtext('text'),
               'template': normalize_thiagos_template(lex_elem
                                                      .findtext('template',
                                                                '')),
               'comment': lex_elem.attrib['comment'],
               'sorted_triples': sorted_sent_triples,
               'references': references
               }

        lexes.append(lex)

    return tuple(lexes)


def load_dataset(dataset_name):

    with open(f'../evaluation/{dataset_name}.pkl', 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def load_test():
    return load_dataset('test')


def load_shared_task_test():

    with open('../evaluation/test_shared_task.pkl', 'rb') as f:
        test = pickle.load(f)

    return test


def load_train():
    return load_dataset('train')


def load_dev():
    return load_dataset('dev')


def make_dataset_pkl(dataset_name):

    filepaths = glob.glob(os.path.join(V_15_BASEPATH,
                                       f'{dataset_name}/**/*.xml'),
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

            entries.append(Entry(eid,
                                 category,
                                 triples,
                                 lexes,
                                 entity_map))

    with open(f'../evaluation/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(entries, f)


def make_test_pkl():
    make_dataset_pkl('test')


def make_train_pkl():
    make_dataset_pkl('train')


def make_dev_pkl():
    make_dataset_pkl('dev')


def make_shared_task_test_pkl():

    entries = []

    tree = ET.parse('../evaluation/testdata_with_lex.xml')
    root = tree.getroot()

    for entry_elem in root.iter('entry'):

        eid = entry_elem.attrib['eid']
        category = entry_elem.attrib['category']
        triples = extract_triples(entry_elem)

        entries.append(Entry(eid, category, triples, None, None))

    with open('../evaluation/test_shared_task.pkl', 'wb') as f:
        pickle.dump(entries, f)


# Legado

def extract_refs_old(dataset):

    import spacy

    nlp = spacy.load('en_core_web_lg')

    name_db = defaultdict(lambda: Counter())
    pronoun_db = defaultdict(lambda: Counter())

    for e in dataset:
        good_lexes = [l for l in e.lexes
                      if l['comment'] == 'good' and e.entity_map]
        for l in good_lexes:
            lexicals = get_lexicalizations(l['text'],
                                           l['template'],
                                           e.entity_map)

            if lexicals:
                for lex_key, lex_values in lexicals.items():
                    for lex_value in lex_values:

                        doc = nlp(lex_value)

                        if len(doc) == 1 and doc[0].pos_ == 'PRON':

                            pronoun_db[lex_key][lex_value] += 1
                        else:
                            name_db[lex_key][lex_value] += 1

    return dict(name_db), dict(pronoun_db)


def lexicalization_match(s, t, entity_map):

    # permite capturar entidades que aparecem mais de uma vez no template
    # ex:
    #   AGENT-1 comeu PATIENT-1 e AGENT-1 vai trabalhar.
    #   João comeu carne e ele vai trabalhar.
    # como uso um regex com grupos nomeados de acordo com o label da entidade
    #    e só posso ter um grupo por label, é preciso criar labels diferentes
    #    para AGENT-1
    # para tanto, adiciono um contado à frente do label, ficando:
    # (?P<AGENT_1_1>.*?) comeu (?P<PATIENT_1_1>.*?) e (?P<AGENT_1_2>.*?)
    # vai trabalhar.

    # contador, por label, de quantos grupos regex já foram utilizados
    lex_counts = Counter()

    def replace_sop(m):

        entity = m.group(0)
        lex_counts[entity] += 1

        # cria o nome do grupo a partir do label
        # como a string está escapada, os labels são recebidos como, ex:
        #    AGENT\\-1
        # devendo ser transformado, para ser utilizado como grupo em:
        #    AGENT_1_1
        # o que ocorre em dois passos:
        #    .translate(TRANS_ESCAPE_TO_RE) - remove \ e troca - por _
        #    '{}_{}'.format(_, lex_counts[entity]) - add o contador de entidade
        entity_group = '{}_{}'.format(entity.translate(TRANS_ESCAPE_TO_RE),
                                      lex_counts[entity])

        # retorna então o regex do grupo que irá capturar
        #    os caracteres da entidade
        return '(?P<{v}>.*?)'.format(v=entity_group)

    # substitui os labels de entidades por regex para capturar suas substrings
    #    adiciona ^ e $ para delimitar o início e fim da string
    t_re = '^{}$'.format(RE_MATCH_TEMPLATE_KEYS_LEX.sub(replace_sop,
                                                        re.escape(t)))

    m = re.match(t_re, s)

    return m


def get_lexicalizations(s, t, entity_map):

    m = lexicalization_match(s, t, entity_map)

    lexicals = defaultdict(list)

    if m:
        for g_name, v in m.groupdict().items():

            entity_label = g_name[:-2].replace('_', '-')
            lex_key = entity_map[entity_label]

            lexicals[lex_key].append(v.lower())

    return dict(lexicals)