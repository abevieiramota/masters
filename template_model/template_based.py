from collections import namedtuple
from more_itertools import flatten
from util import preprocess_so
import re


RE_FIND_SLOT_DEF = re.compile((r'\{(?P<slot_name>slot\d)\-(?P<slot_pos>\d)\}'))


SLOT_NAME = 'slot{}'

Triple = namedtuple('Triple', ['subject', 'predicate', 'object'])


def abstract_triples(triples, slot_template=SLOT_NAME):

    map_entity_id = {}
    abstracted_triples = []
    available_id = 0

    for t in triples:

        if t.subject not in map_entity_id:
            map_entity_id[t.subject] = slot_template.format(available_id)
            available_id += 1

        if t.object not in map_entity_id:
            map_entity_id[t.object] = slot_template.format(available_id)
            available_id += 1

        abstracted_t = Triple(map_entity_id[t.subject],
                              t.predicate,
                              map_entity_id[t.object])

        abstracted_triples.append(abstracted_t)

    return tuple(abstracted_triples)


class JustJoinTemplate:

    def __init__(self):

        self.template_text = '{AGENT-1-1} {p} {PATIENT-1-1}.'
        # FIXME: adicionado apenas para fazer funcionar o len(t.template_triples)
        #    para calcular o tamanho do template
        self.slots = [('AGENT-1', 1), ('PATIENT-1', 1)]

    def fill(self, reg_data, triples):

        if len(triples) != 1:
            raise ValueError(f'This template only accepts data w/ 1 triple.'
                             f' Passed {triples}')

        t = triples[0]

        s = reg_data['AGENT-1-1']
        p = preprocess_so(t.predicate)
        o = reg_data['PATIENT-1-1']

        return f'{s} {p} {o}.'

    def align(self, triples):

        return {'AGENT-1': triples[0].subject,
                'PATIENT-1': triples[0].object}

    def __repr__(self):
        return 'template {s} {p} {o}.'


class TemplateDatabase:

    def __init__(self, template_db_data, template_fallback):

        self.template_db_data = template_db_data 
        self.categories = set(c for (c, _) in template_db_data.keys())
        self.template_fallback = [template_fallback]

    def select(self, category, triples):
        
        a_triples = abstract_triples(triples)
        c_key = (category, a_triples)

        if c_key in self.template_db_data:
            ts = self.template_db_data[c_key]
        else:
            ts = list(flatten(self.template_db_data.get((c, a_triples), [])
                              for c in self.categories))

        if ts:
            return ts 
        elif len(triples) == 1:
            return self.template_fallback
        else:
            return []


class Template:

    def __init__(self, triples, template_text):

        self.template_triples = tuple(triples)
        self.template_text = template_text
        # pairs of (slot-name, slot-n-occurrence)
        self.slots = RE_FIND_SLOT_DEF.findall(self.template_text)

    def fill(self, reg_data, triples=None):

        return self.template_text.format(**reg_data)

    def align(self, triples):
        # retorna um map de {slot_id: entity_id}
        #    onde slot_id pertence a self.template_triples
        #    e entity_id pertence a triples

        positioned_data = {}

        for tt, it in zip(self.template_triples, triples):

            if tt.subject not in positioned_data:

                positioned_data[tt.subject] = it.subject

            if tt.object not in positioned_data:

                positioned_data[tt.object] = it.object

        return positioned_data

    def __hash__(self):

        return hash((self.template_triples, self.template_text))

    def __eq__(self, other):

        return isinstance(self, type(other)) and \
               self.template_triples == other.template_triples and \
               self.template_text == other.template_text

    def __repr__(self):

        return 'Structure: {}\nText: {}'.format(self.template_triples,
                                                self.template_text)
