import re
from template_based import Template, Triple, abstract_triples
from collections import defaultdict, Counter


RE_FIND_THIAGO_SLOT = re.compile('((?:AGENT-.)|(?:PATIENT-.)|(?:BRIDGE-.))')

RE_MATCH_TEMPLATE_KEYS = re.compile(r'(AGENT\\-\d|PATIENT\\-\d|BRIDGE\\-\d)')
TRANS_ESCAPE_TO_RE = str.maketrans('-', '_', '\\')

RE_SPACE_BEFORE_COMMA_DOT = re.compile(r'(?<=\b)\s(?=\.|,)')

RE_WEIRD_QUOTE_MARKS = re.compile(r'(`{1,2})|(\'{1,2})')

# {{}} -> é só para escapar as chaves
SLOT_PLACEHOLDER = '{{{}}}'


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

    delexicalized_triples = delexicalize_triples(sorted_triples, r_entity_map)
    abstracted_triples = abstract_triples(sorted_triples)

    for abstracted_t, delexicalized_t in zip(abstracted_triples,
                                             delexicalized_triples):

        template_text = template_text.replace(delexicalized_t.subject,
                                              SLOT_PLACEHOLDER.format(
                                                      abstracted_t.subject)
                                              )
        template_text = template_text.replace(delexicalized_t.object,
                                              SLOT_PLACEHOLDER.format(
                                                      abstracted_t.object)
                                              )

    # removes @ -> looks like an error
    template_text = template_text.replace('@', '')
    t = Template(abstracted_triples, template_text)

    return t


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
                    sorted_sent_triples.append(sorted_triples)

        lex = {'text': lex_elem.findtext('text'),
               'template': normalize_thiagos_template(lex_elem
                                                      .findtext('template',
                                                                '')),
               'comment': lex_elem.attrib['comment'],
               'sorted_triples': sorted_sent_triples
               }

        lexes.append(lex)

    return tuple(lexes)
