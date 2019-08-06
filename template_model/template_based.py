from collections import namedtuple
from itertools import islice

MSG_ERROR_ROUTE_MATCH = 'structures must be isomorphic'
MSG_ERROR_FROM_TRIPLES = 'triples must contain only one root: {}'

Slot = namedtuple('Slot', ['value', 'predicates'])
Predicate = namedtuple('Predicate', ['value', 'objects'])


class MoreThanOneRootException(Exception):

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def p_slot(s, level=1):

    ident = '\t'*level

    if len(s.predicates) == 0:
        return '{}'.format(s.value)
    else:
        preds_strs = []
        for p in s.predicates:

            pred_str = '\n{}{}'.format(ident, p_pred(p, level+1))
            preds_strs.append(pred_str)

        pred_str = '\n' + ','.join(preds_strs)

        return '[{}, {}]'.format(s.value, pred_str)


def p_pred(p, level):

    ident = '\t'*level

    objs_strs = []
    if len(p.objects) == 1 and len(p.objects[0].predicates) == 0:
        objs_str = '{}'.format(p.objects[0].value)
    else:
        for o in p.objects:

            objs_str = '\n{}{}'.format(ident, p_slot(o, level+1))
            objs_strs.append(objs_str)

        objs_str = ','.join(objs_strs)

    return '<{}, [{}]>'.format(p.value, objs_str)


def to_str(s, lexicalization_f):

    strs = [s.value]

    predicates = list(s.predicates)

    for p in predicates:

        strs.append(p.value)

        for o in p.objects:

            strs.append(o.value)

            predicates.extend(o.predicates)

    return ' '.join((lexicalization_f(x) for x in strs))


# TODO: find a better way to index a structure
def route_predicates(s):

    predicates = [p for p in s.predicates]

    for p in predicates:

        yield p.value

        for o in p.objects:

            predicates.extend(o.predicates)


def route_match(s1, s2):
    # matches two structures returning pairs mapping of
    #    s1.slots.value into s2.slots.value
    # if they have the same structure

    if s1 is None or s2 is None or len(s1.predicates) != len(s2.predicates):
        raise ValueError(MSG_ERROR_ROUTE_MATCH)

    yield s1.value, s2.value

    predicates = list(zip(s1.predicates, s2.predicates))

    for p1, p2 in predicates:

        if p1.value != p2.value:
            raise ValueError(MSG_ERROR_ROUTE_MATCH)

        if len(p1.objects) != len(p2.objects):
            raise ValueError(MSG_ERROR_ROUTE_MATCH)

        for o1, o2 in zip(p1.objects, p2.objects):

            yield o1.value, o2.value

            if len(o1.predicates) != len(o2.predicates):
                raise ValueError(MSG_ERROR_ROUTE_MATCH)

            predicates.extend(zip(o1.predicates, o2.predicates))


class Structure:

    def __init__(self, head):

        self.head = head
        self._predicates = tuple(route_predicates(self.head))

    def position_data(self, data):

        return dict(route_match(self.head, data.head))

    def __hash__(self):

        return hash(self._predicates)

    def __eq__(self, other):

        if isinstance(self, type(other)):
            try:
                self.position_data(other)
                return True
            except ValueError:
                return False
        else:
            return False

    def __repr__(self):

        return p_slot(self.head)

    def __len__(self):

        if not hasattr(self, '__len'):

            ps = self.head.predicates[::]

            i = 0
            for p in ps:

                for o in p.objects:

                    ps.extend(o.predicates)

                i += len(p.objects)

            self.__len = i

        return self.__len

    @staticmethod
    def from_triples(triples):

        slots = {}
        predicates = {}
        subs = set()
        objs = set()

        for t in triples:

            subs.add(t['subject'])
            objs.add(t['object'])

            if t['subject'] in slots:
                s = slots[t['subject']]
            else:
                s = Slot(t['subject'], [])
                slots[t['subject']] = s

            if t['object'] in slots:
                o = slots[t['object']]
            else:
                o = Slot(t['object'], [])
                slots[t['object']] = o

            if (t['subject'], t['predicate']) in predicates:
                p = predicates[(t['subject'], t['predicate'])]

                p.objects.append(o)

            else:
                p = Predicate(t['predicate'], [o])
                s.predicates.append(p)

                predicates[(t['subject'], t['predicate'])] = p

        # gets the slot that isn't object
        subs_not_objs = subs - objs

        # TODO: this must be fixed -> understand better what's going on
        if len(subs_not_objs) != 1:
            raise MoreThanOneRootException(
                    MSG_ERROR_FROM_TRIPLES.format(triples))

        head = slots[list(subs_not_objs)[0]]

        s = Structure(head)

        if len(s) != len(triples):
            raise MoreThanOneRootException(
                    MSG_ERROR_FROM_TRIPLES.format(triples))

        return s


class Template:

    def __init__(self, structure, template_text, meta):

        self.structure = structure
        self.template_text = template_text
        self.meta = meta

    def fill(self, data, lexicalization_f, category, ctx):

        positioned_data = self.structure.position_data(data)

        positioned_data = {k: lexicalization_f(v, category, ctx) for k, v in
                           positioned_data.items()}

        return self.template_text.format(**positioned_data)

    def __hash__(self):

        return hash((self.structure, self.template_text))

    def __eq__(self, other):

        return isinstance(self, type(other)) and \
               self.structure == other.structure and \
               self.template_text == other.template_text

    def __repr__(self):

        return 'Structure: {}\nText: {}'.format(self.structure,
                                                self.template_text)


class StructureData:

    def __init__(self, template_db, fallback_template):
        self.template_db = template_db
        self.fallback_template = fallback_template
        self.template_usage = []

    def get_template(self, s):

        if s in self.template_db:

            if s.head.value in self.template_db[s]:

                return (s, self.template_db[s][s.head.value])
            else:

                return (s, list(islice(self.template_db[s].values(), 0, 1))[0])
        else:
            return None

    def structure(self, entry):

        from collections import deque

        ss_tt = []
        triples = deque(entry['triples'])

        without_fallback = True

        while triples:

            try:
                s = Structure.from_triples(triples)
                s_t = self.get_template(s)
            except MoreThanOneRootException:
                s_t = None

            if s_t is not None:

                ss_tt.append(s_t)
                triples.clear()
            else:
                t = triples.pop()

                s = Structure.from_triples([t])

                s_t = self.get_template(s)

                if s_t:
                    ss_tt.append(s_t)
                else:
                    ss_tt.append((s, self.fallback_template))
                    without_fallback = False

        self.template_usage.append(without_fallback)

        return ss_tt[::-1]


class JustJoinTemplate:

    def fill(self, data, lexicalization_f, category, ctx):

        if len(data) != 1:
            raise ValueError(f'This template only accepts data w/ 1 triple.'
                             f' Passed {data}')

        s = lexicalization_f(data.head.value, category, ctx)
        p = lexicalization_f(data.head.predicates[0].value, category, ctx)
        o = lexicalization_f(data.head.predicates[0].objects[0].value,
                             category,
                             ctx)

        return f'{s} {p} {o}.'

    def __repr__(self):
        return 'template {s} {p} {o}.'


class SelectTemplate:

    def select_template(self, structured_data):

        return [(s, ts.most_common(1)[0][0]) for s, ts in structured_data]


class MakeText:

    def __init__(self, lexicalization_f=None):
        self.lexicalization_f = lexicalization_f

    def make_text(self, lexicalized_templates, e):

        texts = []
        ctx = {'seen': set()}
        for s, t in lexicalized_templates:

            text = t.fill(s, self.lexicalization_f, e['category'], ctx)

            texts.append(text)

        return ' '.join(texts)