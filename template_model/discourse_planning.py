# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from itertools import permutations
from reading_thiagos_templates import RE_FIND_THIAGO_SLOT
from math import ceil


class NaiveDiscoursePlanFeature:

    def __init__(self):

        self._counters = defaultdict(lambda: Counter())
        self._all_counts = defaultdict(int)
        self._vocabs = defaultdict(set)

    def fit(self, triplesets, counts):

        for tripleset, count in zip(triplesets, counts):

            seq_len = len(tripleset)

            seq = [t.predicate for t in tripleset]

            padded_seq = ['<INI>'] + seq + ['<END>']

            for pair in zip(padded_seq[:-1], padded_seq[1:]):

                self._counters[seq_len][pair] += count
                self._counters[seq_len][pair[0]] += count
                self._counters[seq_len][pair[1]] += count
                self._vocabs[seq_len].add(pair[0])
                self._vocabs[seq_len].add(pair[1])

    def extract(self, triples):

        prob = 1
        seq = [t.predicate for t in triples]
        seq_len = len(seq)

        padded_seq = ['<INI>'] + seq + ['<END>']

        for pair in zip(padded_seq[:-1], padded_seq[1:]):

            prob_pair = (self._counters[seq_len][pair] + 1) / \
                        (self._counters[seq_len][pair[0]] +
                         len(triples))

            prob *= prob_pair

        return {'feature_plan_naive_discourse_prob': prob}

    def sort(self, l_triples):

        return sorted(l_triples,
                      key=lambda t:
                      self.extract(t)['feature_plan_naive_discourse_prob'])


class DiscoursePlanning:

    def __init__(self, pct, feature_extractors=None, sort=None):

        if not feature_extractors:
            feature_extractors = []

        self.feature_extractors = feature_extractors
        self.sort = sort if sort else lambda x, y: list(x)
        self.pct=pct

    def plan(self, e):

        plans = list(self.sort(permutations(e.triples), e))

        n_max = ceil(self.pct * len(plans))

        for i, plan in zip(range(n_max), plans):

            plan_f = {'plan': plan,
                      'feature_plan_is_first': i == 0}

            for fe in self.feature_extractors:

                plan_f.update(fe.extract(plan))

            yield plan_f


from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from itertools import product, combinations
from template_based2 import abstract_triples
from functools import reduce
from scipy.stats import kendalltau
from operator import itemgetter
from sklearn.model_selection import train_test_split


def calc_kendall(o1, good_os):

    all_kendall = [kendalltau(o, o1).correlation for o in good_os]

    max_kendall = max(all_kendall)

    return max_kendall


def extract_orders(e):

    orders = Counter()

    for lexe in e.lexes:

        slots = RE_FIND_THIAGO_SLOT.findall(lexe['template'])
        positions = {}
        for i, k in enumerate(slots):
            if k not in positions:
                positions[k] = i
        positions = defaultdict(lambda: 10000, positions)
        sorted_triples = tuple(sorted(e.triples,
                                      key=lambda t:
                                          positions[e.r_entity_map[t.object]]))

        orders[sorted_triples] += 1

    return orders


def frac_siblings_subjects(o):

    i = 0
    current_s = None
    for o_ in o:
        if o_.subject == current_s:
            i += 1
        else:
            current_s = o_.subject

    subs = Counter(o_.subject for o_ in o)
    max_siblings = sum(v - 1 for v in subs.values())

    return (i + 1) / (max_siblings + len(o))


def frac_siblings_objects(o):

    i = 0
    current_o = None
    for o_ in o:
        if o_.object == current_o:
            i += 1
        else:
            current_o = o_.object

    obs = Counter(o_.object for o_ in o)
    max_siblings = sum(v - 1 for v in obs.values())

    return (i + 1) / (max_siblings + len(o))


def frac_siblings_predicates(o):

    i = 0
    current_p = None
    for o_ in o:
        if o_.predicate == current_p:
            i += 1
        else:
            current_p = o_.predicate

    preds = Counter(o_.predicate for o_ in o)
    max_siblings = sum(v - 1 for v in preds.values())

    return (i + 1) / (max_siblings + len(o))


def frac_chains(o):

    n = 0
    for t, t_1 in zip(o[:-1], o[1:]):
        if t.object == t_1.subject:
            n += 1

    subs = set(t.subject for t in o)
    objs = set(t.object for t in o)

    len_intersect = len(subs.intersection(objs))

    return (n + 1) / (len_intersect + len(o))


def is_first_the_main(o):

    subjs = set(t.subject for t in o)
    objs = set(t.object for t in o)

    s_not_o = subjs - objs

    return o[0].subject in s_not_o


class ExtractFeatures(TransformerMixin):

    def fit(self, X, y=None):

        self.freq_all = Counter()
        self.freq_bigrams = Counter()
        self.freq_unigrams = Counter()
        self.size_training = Counter()

        good_orders = [x for i, x in enumerate(X) if y[i] == 1.0]

        for o in good_orders:
            for pos, t in enumerate(o):
                self.freq_unigrams[t.predicate] += 1

            a_o = abstract_triples(o)

            self.freq_all[a_o] += 1

            for t1, t2 in zip(o[:-1], o[1:]):
                self.freq_bigrams[(t1.predicate, t2.predicate)] += 1

            self.size_training[len(o)] += 1

        self.feature_names_ = ['freq_all',
                               'frac_chains',
                               'is_first_the_main',
                               'frac_siblings_subjects',
                               'frac_siblings_objects',
                               'frac_siblings_predicates',
                               'naive_prob_predicates']

        return self

    def transform(self, X, y=None):

        return [self.extract_features(o) for o in X]

    def extract_features(self, order):

        a_o = abstract_triples(order)

        naive_prob_predicates = reduce(lambda x, y: x*y,
                                       [(self.freq_bigrams[(t1.predicate, t2.predicate)] + 1) / (self.freq_unigrams[t1.predicate] + len(order))
                                        for t1, t2 in zip(order[:-1], order[1:])])

        features = [self.freq_all[a_o] / self.size_training[len(order)],
                    frac_chains(order),
                    is_first_the_main(order),
                    frac_siblings_subjects(order),
                    frac_siblings_objects(order),
                    frac_siblings_predicates(order),
                    naive_prob_predicates]

        return features


from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


def get_pipeline():

    pipeline = Pipeline([
        ('mms',  MinMaxScaler()),
        ('clf', Lasso(alpha=0.0001))
    ])

    return pipeline


def get_sorter(models, fe):

    def sort(orders, e):

        len_triples = len(e.triples)

        if len_triples == 1:
            return orders

        orders = list(orders)

        data = fe[len_triples].transform(orders)

        scores = models[len_triples].predict(data)

        return [orders[i] for i, s in sorted(enumerate(scores),
                                             key=lambda x: x[1],
                                             reverse=True)]

    return sort


class DiscoursePlanningSorter:

    def fit(self, entries):

        self.n_triples = set(len(e.triples) for e in entries) - {1}
        self.models = {}
        self.sorters = {}

        for n_triple in self.ntriples:

            selected_entries = [e for e in entries if len(e.triples) == n_triple]

            X, y = make_data(selected_entries)

            model = get_pipeline()

            model.fit(X, y)

            self.models[n_triple] = model
            self.sorters[n_triple] = get_sorter(model)

            print(f'Trained ({n_triple})')

    def sort(self, orders, e):

        n_triples = len(orders)

        if n_triples == 1:
            return orders

        if n_triples not in self.sorters:
            return orders

        return self.sorters[n_triples](orders, e)



