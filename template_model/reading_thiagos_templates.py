import re
from template_based import Template, Triple, abstract_triples
from collections import defaultdict, Counter, namedtuple
from nltk import sent_tokenize
import pickle
import glob
import xml.etree.ElementTree as ET
import os
from preprocessing import *
from more_itertools import flatten


# usado na extração de expressões de referência
#    alinhando texto e template
#    tem \\ porque é aplicado sobre o template após um re.escape
RE_MATCH_TEMPLATE_KEYS_LEX = re.compile((r'(AGENT\\-\d|PATIENT\\-\d'
                                         r'|BRIDGE\\-\d)'))
RE_MATCH_TEMPLATE_KEYS = re.compile(r'(AGENT\-\d|PATIENT\-\d|BRIDGE\-\d)')
TRANS_ESCAPE_TO_RE = str.maketrans('-', '_', '\\')

# {{}} -> é só para escapar as chaves
SLOT_PLACEHOLDER = '{{{}-{}}}'

V_15_BASEPATH = '../../webnlg/data/v1.5/en/'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


Entry = namedtuple('Entry', ['eid',
                             'category',
                             'triples',
                             'lexes',
                             'entity_map'])


def extract_templates(dataset):

    entries_templates = []
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
                               e.entity_map)

            if ts:
                lexes_templates.append((l, ts))

        if lexes_templates:
            entries_templates.append((e, lexes_templates))

    return entries_templates


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


def make_template(sorted_triples, template_text, entity_map):

    templates = []

    # apenas tokeniza com sent_tokenize se o particionamento das triplas
    #    indicar que há mais de uma sentença
    #    reduz problemas com tokenização de sentenças errado
    if len(sorted_triples) == 1:
        template_sens = [template_text]
    else:
        template_sens = sent_tokenize(template_text)

    # se a quantidade de partes de triplas é diferente da quantidade de sentenças, retorna vazio
    if len(sorted_triples) != len(template_sens):
        return []

    for template_sen, triples_sen in zip(template_sens, sorted_triples):

        # triplas em que os sujeito/objeto foram substituídos pelos identificadores de slot (PATIENT-1 etc)
        delexicalized_triples = delexicalize_triples(triples_sen, entity_map)
        # triplas em que os sujeito/objeto foram substituídos por slots (slot-0, slot-1 etc), de acordo com a modelagem da solução
        abstracted_triples = abstract_triples(triples_sen)

        assert len(delexicalized_triples) == len(abstracted_triples)

        # map para fazer a tradução de um slot na modelagem do Tiago para a modelagem da solução
        #    ex: PATIENT-1 -> slot-2
        map_template_key_to_slot = {}
        for d, a in zip(delexicalized_triples, abstracted_triples):
            map_template_key_to_slot[d.subject] = a.subject
            map_template_key_to_slot[d.object] = a.object

        # contador responsável por contar quantas ocorrências de uma entidade já foram vistas
        entity_ref_counter = Counter()

        # extrai as ocorrências de slots na modelagem do Tiago nos templates
        keys_in_template_sen = RE_MATCH_TEMPLATE_KEYS.findall(template_sen)

        # filtro para remover casos em que a quantidade de slots é maior que a de sujeitos/objetos
        if len(keys_in_template_sen) > 2*len(triples_sen):
            templates.append(None)
            continue

        # se houver algum slot no template que não está mapeado, também filtra
        if any(k not in map_template_key_to_slot for k in keys_in_template_sen):
            templates.append(None)
            continue

        # para cada slot encontrado no template do Tiago
        for key in keys_in_template_sen:
            # quantidade de vezes que essa entidade já foi referenciada
            i_occurrence = entity_ref_counter[key]
            # slot na minha modelagem, identificando uma entidade ex: slot-0
            slot = map_template_key_to_slot[key]
            # slot na minha modelagem, identificando uma entidade e quantas vezes já foi referenciada ex: slot-0-1
            s_slot = SLOT_PLACEHOLDER.format(slot, i_occurrence)
            # substitui no template o slot do Tiago pelo meu slot
            template_sen = template_sen.replace(key, s_slot, 1)
            # incrementa o contador de referências da entidade
            entity_ref_counter[key] += 1

        # slots existentes
        slots = set() 
        for t in abstracted_triples:
            slots.add(t.subject)
            slots.add(t.object)

        # checa se todos os slots aparecem no template
        if all(slot in template_sen for slot in slots):
            # removes @ -> looks like an error
            template_sen = preprocess_text(template_sen)
            t = Template(abstracted_triples, template_sen)

            templates.append(t)
        else:
            templates.append(None)

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
               'template': normalize_thiagos_template(lex_elem.findtext('template', '')),
               'comment': lex_elem.attrib['comment'],
               'sorted_triples': sorted_sent_triples,
               'references': references
               }

        lexes.append(lex)

    return tuple(lexes)


def extract_shared_task_lexes(entry_elem):

   lexes = []

   for lex_elem in entry_elem.findall('lex'):
      lex = {'text': lex_elem.text}
      lexes.append(lex)
   return tuple(lexes)


def load_datasets(db_names):

    return list(flatten(load_dataset(db_name) for db_name in db_names))


def load_dataset(db_name):

    with open(os.path.join(BASE_DIR, f'../evaluation/{db_name}.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def load_test():
    return load_dataset('test')


def load_shared_task_test():

    with open(os.path.join(BASE_DIR,
                           '../evaluation/test_shared_task.pkl'), 'rb') as f:
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

    with open(os.path.join(BASE_DIR,
                           f'../evaluation/{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(entries, f)


def make_test_pkl():
    make_dataset_pkl('test')


def make_test_seen_pkl():

    # a base test contém as entries extraídas da base do Thiago,
    #   mas em ordem diferente da da competição
    # já a shared task test contém as entries na ordem utilizada na competição
    #   o subconjunto seen são as primeiras 970 entries da base test da competição
    # todo esse código tem como finalidade selecionar na base do Thiago
    #   as entries do subset seen, olhando quais são seus eid e categoria
    #   na base shared task test
    test = load_test()
    test_shared_task = load_shared_task_test()

    test_seen = test_shared_task[:970]

    eids_categories_seen = {(e.eid, e.category) for e in test_seen}

    test_seen_full = [e for e in test
                      if (e.eid, e.category) in eids_categories_seen]

    with open(os.path.join(BASE_DIR,
                           f'../evaluation/test_seen.pkl'), 'wb') as f:
        pickle.dump(test_seen_full, f)


def make_train_pkl():
    make_dataset_pkl('train')


def make_dev_pkl():
    make_dataset_pkl('dev')


def make_shared_task_test_pkl():

    entries = []

    tree = ET.parse(os.path.join(BASE_DIR,
                                 '../evaluation/testdata_with_lex.xml'))
    root = tree.getroot()

    for entry_elem in root.iter('entry'):

        eid = entry_elem.attrib['eid']
        category = entry_elem.attrib['category']
        triples = extract_triples(entry_elem)
        lexes = extract_shared_task_lexes(entry_elem)

        entries.append(Entry(eid, category, triples, lexes, None))

    with open(os.path.join(BASE_DIR,
                           '../evaluation/test_shared_task.pkl'), 'wb') as f:
        pickle.dump(entries, f)


def lexicalization_match(s, t):

    s = normalize_thiagos_template(s)

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

    m = lexicalization_match(s, t)

    lexicals = defaultdict(list)

    if m:
        for g_name, v in m.groupdict().items():

            entity_label = g_name[:-2].replace('_', '-')
            lex_key = entity_map[entity_label]

            lexicals[lex_key].append(v)

    return dict(lexicals)
