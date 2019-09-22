import re
from template_based import Template, Triple, abstract_triples
from collections import defaultdict, Counter, namedtuple
from nltk import sent_tokenize
import pickle
import glob
import xml.etree.ElementTree as ET
import os


RE_FIND_THIAGO_SLOT = re.compile('((?:AGENT-.)|(?:PATIENT-.)|(?:BRIDGE-.))')

RE_MATCH_TEMPLATE_KEYS = re.compile(r'(AGENT\\-\d|PATIENT\\-\d|BRIDGE\\-\d)')
TRANS_ESCAPE_TO_RE = str.maketrans('-', '_', '\\')

RE_SPACE_BEFORE_COMMA_DOT = re.compile(r'(?<=\b)\s(?=\.|,)')

RE_WEIRD_QUOTE_MARKS = re.compile(r'(`{1,2})|(\'{1,2})')

# {{}} -> é só para escapar as chaves
SLOT_PLACEHOLDER = '{{{}}}'

V_15_BASEPATH = '../../webnlg/data/v1.5/en/'


PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')
CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')


Entry = namedtuple('Entry', ['eid',
                             'category',
                             'triples',
                             'lexes',
                             'entity_map',
                             'r_entity_map'])


# it also creates the corpus used to train the template selector
def make_template_db(dataset_name):

    template_db = defaultdict(set)
    template_model_texts = []
    extraction_error = []

    dataset = load_dataset(dataset_name)
    dataset_w_entity_map = [e for e in dataset if e.r_entity_map]

    for e in dataset_w_entity_map:

        good_lexes = [l
                      for l in e.lexes
                      if (
                              l['comment'] == 'good'
                              and l['template']
                              and l['sorted_triples']
                              )
                      ]

        for l in good_lexes:

            ts = make_template(l['sorted_triples'],
                               l['template'],
                               e.r_entity_map)

            if not ts:
                extraction_error.append((e, l))

            for t, triples in zip(ts, l['sorted_triples']):

                template_db[(e.category, t.template_triples)].add(t)

                text = t.fill(triples, lambda x, ctx: x, None)

                template_model_texts.append(text.lower())

    template_db = dict(template_db)

    with open('../data/templates/template_db/tdb', 'wb') as f:
        pickle.dump(template_db, f)

    with open('../data/kenlm/ts_texts.txt', 'w', encoding='utf-8') as f:
        for t in template_model_texts:
            f.write(f'{t}\n')

    return extraction_error


def normalize_thiagos_template(s):
    # removes single space between an entity and a dot or a comma

    s = RE_SPACE_BEFORE_COMMA_DOT.sub('', s)

    return RE_WEIRD_QUOTE_MARKS.sub('"', s)


def delexicalize_triples(triples, r_entity_map):

    delexicalized_triples = []

    for t in triples:

        delexicalized_t = Triple(r_entity_map[t.subject],
                                 t.predicate,
                                 r_entity_map[t.object])

        delexicalized_triples.append(delexicalized_t)

    return delexicalized_triples


def make_template(sorted_triples, template_text, r_entity_map):

    templates = []

    template_sens = sent_tokenize(template_text)

    for template_sen, triples_sen in zip(template_sens, sorted_triples):

        delexicalized_triples = delexicalize_triples(triples_sen, r_entity_map)
        abstracted_triples = abstract_triples(triples_sen)

        for abstracted_t, delexicalized_t in zip(abstracted_triples,
                                                 delexicalized_triples):

            template_sen = template_sen.replace(delexicalized_t.subject,
                                                SLOT_PLACEHOLDER.format(
                                                        abstracted_t.subject)
                                                )
            template_sen = template_sen.replace(delexicalized_t.object,
                                                SLOT_PLACEHOLDER.format(
                                                        abstracted_t.object)
                                                )

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


def get_lexicalizations(s, t, entity_map):

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
    t_re = '^{}$'.format(RE_MATCH_TEMPLATE_KEYS.sub(replace_sop,
                                                    re.escape(t)))

    m = re.match(t_re, s)

    lexicals = defaultdict(list)

    if m:
        for g_name, v in m.groupdict().items():

            entity_label = g_name[:-2].replace('_', '-')
            lex_key = entity_map[entity_label]

            lexicals[lex_key].append(v.lower())

    return dict(lexicals)


def extract_triples(entry_elem):

    triples = []

    modifiedtripleset_elem = entry_elem.find('modifiedtripleset')

    for t in modifiedtripleset_elem.findall('mtriple'):

        sub, pred, obj = [x.strip(' "') for x in t.text.split('|')]

        triple = Triple(sub, pred, obj)

        triples.append(triple)

    return tuple(triples)


def extract_entity_map(entry_elem):

    entity_dict = {}

    for ent_elem in entry_elem.find('entitymap').findall('entity'):

        ent_placeholder, ent_value = ent_elem.text.split('|')

        entity_dict[ent_placeholder.strip()] = ent_value.strip(' "')

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
                    sub, pred, obj = [x.strip(' "') for x in t.text.split('|')]

                    triple = Triple(sub, pred, obj)

                    sorted_triples.append(triple)

                if sorted_triples:
                    sorted_sent_triples.append(tuple(sorted_triples))

        lex = {'text': lex_elem.findtext('text'),
               'template': normalize_thiagos_template(lex_elem
                                                      .findtext('template',
                                                                '')),
               'comment': lex_elem.attrib['comment'],
               'sorted_triples': sorted_sent_triples
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


def preprocess_so(so):

    parenthesis_preprocessed = PARENTHESIS_RE.sub(r'\g<2> \g<1>', so)
    underline_removed = parenthesis_preprocessed.replace('_', ' ')
    camelcase_preprocessed = CAMELCASE_RE.sub(r'\g<1> \g<2>',
                                              underline_removed)

    return camelcase_preprocessed.strip('" ')


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
            r_entity_map = {v: k for k, v in entity_map.items()}

            entries.append(Entry(eid,
                                 category,
                                 triples,
                                 lexes,
                                 entity_map,
                                 r_entity_map))

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

        entries.append(Entry(eid, category, triples, None, None, None))

    with open('../evaluation/test_shared_task.pkl', 'wb') as f:
        pickle.dump(entries, f)


sorted_triples = [(Triple(subject='Serie_A', predicate='champions', object='Juventus_F.C.'),),
   (Triple(subject='A.S._Roma', predicate='league', object='Serie_A'),
    Triple(subject='A.S._Roma', predicate='ground', object='Stadio_Olimpico'))]
template_text = 'PATIENT-1 have been BRIDGE-1 champions. AGENT-1, who "s ground is PATIENT-2 also play in the same league.'
r_entity_map = {'A.S._Roma': 'AGENT-1', 'Serie_A': 'BRIDGE-1', 'Juventus_F.C.': 'PATIENT-1', 'Stadio_Olimpico': 'PATIENT-2'}

make_template(sorted_triples, template_text, r_entity_map)
