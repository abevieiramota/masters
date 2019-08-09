from collections import namedtuple
from itertools import permutations, product
from more_itertools import partitions, flatten


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

    def __init__(self, triples, template_text, meta):

        self.template_triples = tuple(triples)
        self.template_text = template_text
        self.meta = meta

    def fill(self, triples, lexicalization_f, ctx):

        positioned_data = {}

        for tt, it in zip(self.template_triples, triples):

            if tt.subject not in positioned_data:

                positioned_data[tt.subject] = lexicalization_f(it.subject,
                                                               ctx)

            if tt.object not in positioned_data:

                positioned_data[tt.object] = lexicalization_f(it.object,
                                                              ctx)

        return self.template_text.format(**positioned_data)

    def __hash__(self):

        return hash((self.template_triples, self.template_text))

    def __eq__(self, other):

        return isinstance(self, type(other)) and \
               self.template_triples == other.template_triples and \
               self.template_text == other.template_text

    def __repr__(self):

        return 'Structure: {}\nText: {}'.format(self.template_triples,
                                                self.template_text)


class StructureData:

    def __init__(self, template_db, fallback_template):
        self.template_db = template_db
        self.fallback_template = fallback_template
        self.template_usage = []

    def structure(self, triples):

        triples_permutations = permutations(triples)
        all_partitions = flatten([partitions(p) for p in triples_permutations])
        all_partitons = sorted(all_partitions,
                               key=lambda e: [len(e)] +
                               [i*len(z)*-1 for i, z in enumerate(e)])

        all_tt = []

        for triples_partition in all_partitons:

            tt = []
            has_all = True

            for triple_partition in triples_partition:

                a_triples = abstract_triples(triple_partition)

                if a_triples in self.template_db:

                    template = self.template_db[a_triples]\
                                   .most_common()[0][0]

                    tt.append((triple_partition, template))
                else:
                    has_all = False
                    break

            if has_all:

                all_tt.append(tt)

        if all_tt:
            return all_tt

        tt = []
        for t in triples:

            abstracted_t = abstract_triples([t])

            if abstracted_t in self.template_db:

                template = self.template_db[abstracted_t]\
                               .most_common()[0][0]
            else:

                template = self.fallback_template

            tt.append(([t], template))

        all_tt.append(tt)

        return all_tt


class JustJoinTemplate:

    def __init__(self):

        self.template_text = '{s} {p} {o}.'

    def fill(self, triples, lexicalization_f, ctx):

        if len(triples) != 1:
            raise ValueError(f'This template only accepts data w/ 1 triple.'
                             f' Passed {triples}')

        t = triples[0]

        s = lexicalization_f(t.subject, ctx)
        p = lexicalization_f(t.predicate, ctx)
        o = lexicalization_f(t.object, ctx)

        return f'{s} {p} {o}.'

    def __repr__(self):
        return 'template {s} {p} {o}.'


class MakeText:

    def __init__(self, lexicalization_f=None):
        self.lexicalization_f = lexicalization_f

    def make_text(self, all_tt):

        all_texts = []

        for triples_templates in all_tt:

            texts = []
            for triples, t in triples_templates:

                text = t.fill(triples, self.lexicalization_f)

                texts.append(text)

            all_texts.append(' '.join(texts))

        return all_texts


class TemplatePlanModel:

    def __init__(self,
                 discourse_planning,
                 sentence_aggregation,
                 template_selection):

        self.discourse_planning = discourse_planning
        self.sentence_aggregation = sentence_aggregation
        self.template_selection = template_selection

    def _generate_without_ranking(self, entry):

        for plan in self.discourse_planning.plan(entry):
            for agg in self.sentence_aggregation.agg(entry, plan):
                templates = self.template_selection.select(entry, agg)

                if templates is not None:
                    return self.make_text(templates)

        return None

    def _generate_with_ranking(self, entry, ranking, return_all=False):

        all_texts = []

        for plan in self.discourse_planning.plan(entry):
            for agg in self.sentence_aggregation.agg(entry, plan):
                templates = self.template_selection.select(entry, agg)

                texts = self.make_text(templates)

                all_texts.extend(texts)

        if texts:

            ranked = ranking(all_texts)

            if return_all:
                return ranked
            else:
                return ranked[0]
        else:
            return None

    def generate(self, entry, ranking=None, return_all=None):

        if ranking:
            return self._generate_with_ranking(entry, ranking, return_all)
        else:
            return self._generate_without_ranking(entry)

    def make_text(self, templates):

        all_texts = []

        for agg_templates in templates:

            curr_texts = []

            triples = agg_templates[0]

            for template_dict in agg_templates[1]:

                ctx = {'seen': set()}

                template = template_dict['template']

                text = template.fill(triples, self.lexicalization, ctx)

                curr_texts.append(text)

            all_texts.append(curr_texts)

        return [' '.join(comb) for comb in product(*all_texts)]
