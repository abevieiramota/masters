# -*- coding: utf-8 -*-
from discourse_planning import DiscoursePlanning, NaiveDiscoursePlanFeature, make_data, get_pipeline, get_sorter
from sentence_aggregation import SentenceAggregation, LessPartsBiggerFirst
from template_selection import TemplateSelection
from template_based2 import JustJoinTemplate
import pandas as pd
import pickle
from reg import REGer
from collections import defaultdict
from random import shuffle
from model import TextGenerationModel
from util import load_train_dev, load_test


template_db = pd.read_pickle('../data/templates/template_db/template_db')

triples_to_templates = defaultdict(list)

for v in template_db.to_dict(orient='record'):

    triples_to_templates[v['template_triples']].append(v)

triples_to_templates = dict(triples_to_templates)

np = NaiveDiscoursePlanFeature()
np.fit(template_db['template_triples'],
       template_db['feature_template_cnt_in_category'])

with open('../data/templates/lexicalization/thiago_name_db', 'rb') as f:
    name_db = pickle.load(f)

with open('../data/templates/lexicalization/thiago_pronoun_db', 'rb') as f:
    pronoun_db = pickle.load(f)


td = load_train_dev()
test = load_test()


def make_main_model_data():

    td_to_train_discourse_plan_ranker = [t for t in td if len(t.triples) > 1 and t.r_entity_map]

    X, y = make_data(td_to_train_discourse_plan_ranker)

    with open('../data/templates/discourse_plan_data', 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)


def get_model(pct, sorter):

    dp = DiscoursePlanning(pct, [np], sort=sorter)

    # Sentence Aggregation
    lpbpf = LessPartsBiggerFirst()
    sa = SentenceAggregation(triples_to_templates, [lpbpf])

    # Template Selection
    ts = TemplateSelection(triples_to_templates, JustJoinTemplate())

    # REG
    refer = REGer(pronoun_db, name_db).refer

    model = TextGenerationModel(dp, sa, ts, refer)

    return model


def get_random_model(pct):
    # Discourse Planning

    def random_order(orders, e):

        orders_ = list(orders)[::]
        shuffle(orders_)

        return orders_

    return get_model(pct, random_order)



def get_main_model(pct):

    with open('../data/templates/discourse_plan_data', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    pipe = get_pipeline()

    pipe.fit(X, y)

    sorter = get_sorter(pipe)

    return get_model(pct, sorter)
