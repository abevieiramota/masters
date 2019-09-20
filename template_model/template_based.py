from collections import namedtuple
import re


RE_REPLACE_SLOT = re.compile(r'\{slot\d\}')


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


class Template:

    def __init__(self, triples, template_text):

        self.template_triples = tuple(triples)
        self.template_text = template_text

    def fill(self, triples, reg_f, ctx):

        aligned_data = self.align(triples)
        reg_data = {k: reg_f(v, ctx) for k, v in aligned_data.items()}

        return self.template_text.format(**reg_data)

    def align(self, triples):

        positioned_data = {}

        for tt, it in zip(self.template_triples, triples):

            if tt.subject not in positioned_data:

                positioned_data[tt.subject] = it.subject

            if tt.object not in positioned_data:

                positioned_data[tt.object] = it.object

        return positioned_data

    def fill_(self, slots):

        return self.template_text.format(**slots)

    def __hash__(self):

        return hash((self.template_triples, self.template_text))

    def __eq__(self, other):

        return isinstance(self, type(other)) and \
               self.template_triples == other.template_triples and \
               self.template_text == other.template_text

    def __repr__(self):

        return 'Structure: {}\nText: {}'.format(self.template_triples,
                                                self.template_text)


class JustJoinTemplate:

    def __init__(self):

        self.template_text = '{s} {p} {o}.'

    def fill(self, triples, reg_f, ctx):

        if len(triples) != 1:
            raise ValueError(f'This template only accepts data w/ 1 triple.'
                             f' Passed {triples}')

        t = triples[0]

        s = reg_f(t.subject, ctx)
        p = reg_f(t.predicate, ctx)
        o = reg_f(t.object, ctx)

        return f'{s} {p} {o}.'

    def fill_(self, slots):

        return self.template_text.format(s=slots['slot0'],
                                         o=slots['slot1'],
                                         p='')

    def align(self, triples):

        return {'slot0': triples[0].subject,
                'slot1': triples[0].object}

    def __repr__(self):
        return 'template {s} {p} {o}.'
