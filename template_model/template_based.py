from collections import namedtuple
from more_itertools import flatten
import re
from preprocessing import *


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

    def __init__(self, predicate):

        self.template_text = '{slot-0-1} ' + preprocess_so(predicate) + ' {slot-1-1}.'
        # FIXME: adicionado apenas para fazer funcionar o len(t.template_triples)
        #    para calcular o tamanho do template
        self.slots = [('slot-0', 1), ('slot-1', 1)]
        self.template_triples = [Triple('None', predicate, 'None')]

    def fill(self, refs):

        if len(refs) != 2:
            raise ValueError(f'This template only accepts data w/ 1 triple. Passed {refs}')

        reg_data = dict(zip(self.slots_placeholders, refs))

        return self.template_text.format(**reg_data)

    @property
    def slots_placeholders(self):
        return (f'{slot_name}-{slot_pos}' for slot_name, slot_pos in self.slots)

    def align(self, triples):

        return {'slot-0': triples[0].subject,
                'slot-1': triples[0].object}

    def __repr__(self):
        return 'template {s} {p} {o}.'


class TemplateDatabase:

    def __init__(self, template_db_data, template_fallback):

        self.template_db_data = template_db_data 
        self.template_fallback = template_fallback

    def select(self, triples):
        
        a_triples = abstract_triples(triples)

        ts = self.template_db_data.get(a_triples, [])

        if ts:
            return ts 
        elif len(triples) == 1:
            return [self.template_fallback(triples[0].predicate)]
        else:
            return []


class Template:

    def __init__(self, triples, template_text):

        self.template_triples = tuple(triples)
        self.template_text = template_text
        # pairs of (slot-name, slot-n-occurrence)
        self.slots = RE_FIND_SLOT_DEF.findall(self.template_text)

    def fill(self, refs):

        reg_data = dict(zip(self.slots_placeholders, refs))

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

    @property
    def slots_placeholders(self):
        return (f'{slot_name}-{slot_pos}' for slot_name, slot_pos in self.slots)

    def __hash__(self):

        return hash((self.template_triples, self.template_text))

    def __eq__(self, other):

        return isinstance(self, type(other)) and \
               self.template_triples == other.template_triples and \
               self.template_text == other.template_text

    def __repr__(self):

        return 'Structure: {}\nText: {}'.format(self.template_triples,
                                                self.template_text)
