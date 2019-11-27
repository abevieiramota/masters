# -*- coding: utf-8 -*-

def load_gold():

    with open('reg_db.csv', 'r', encoding='utf-8') as f:

        f.readline()

        refs = []

        for l in f.readlines():

            data = l[:-1].split('\t')

            refs.append(data[-1])

        return refs

def seen_test():

    from reading_thiagos_templates import load_shared_task_test, load_test

    stest = load_shared_task_test()
    test = load_test()

    with open('../evaluation/subsets/test/old-cat.txt', 'r') as f:

        seen_ids = [int(i[:-1]) for i in f.readlines()]

    seen_stest = [stest[i] for i in seen_ids]

    test = {(e.eid, e.category): e for e in test}

    return [test[(e.eid, e.category)] for e in seen_stest]


def make_db():
    test = seen_test()

    data = []

    for i, e in enumerate(test):

        for l in e.lexes:

            if l['references']:

                for r in l['references']:

                    data.append((e.eid, e.category, r['entity'], r['ref'].lower()))


    with open('reg_db.csv', 'w', encoding='utf-8') as f:

        f.write('eid\tcategory\tentity\tref\n')

        for d in data:

            f.write('{}\t{}\t{}\t{}\n'.format(*d))


def evaluate(refs):

    from reading_thiagos_templates import normalize_thiagos_template

    with open('reg_db.csv', 'r', encoding='utf-8') as f:

        f.readline()

        refs_ok = []

        for l in f:

            refs_ok.append(l.split('\t')[-1][:-1])

    return sum(normalize_thiagos_template(ref) == normalize_thiagos_template(ref_ok)
               for ref, ref_ok in zip(refs, refs_ok)) / len(refs_ok)


def naive():

    from util import preprocess_so

    with open('reg_db.csv', 'r', encoding='utf-8') as f:

        f.readline()

        refs = []

        for l in f:

            refs.append(preprocess_so(l.split('\t')[-2].lower()))

    return refs


def most_freq_fallto_naive():

    from pretrained_models import load_abe_referrer_counters

    reg = load_abe_referrer_counters(('train', 'dev'))

    from util import preprocess_so

    with open('reg_db.csv', 'r', encoding='utf-8') as f:

        f.readline()

        refs = []

        ent_db = []
        ent_naive = []

        for i, l in enumerate(f):

            data = l[:-1].split('\t')

            ent = data[2]

            naive = preprocess_so(data[-2].lower())

            if ent in reg['1st']:
                c = reg['1st'][ent]

                refs.append(c.most_common()[0][0])
                ent_db.append(ent)
            else:
                refs.append(naive)
                ent_naive.append(ent)

    return refs, ent_db, ent_naive


def artefact():

    from util import preprocess_so
    from reading_thiagos_templates import load_test, load_train, load_dev
    from more_itertools import flatten

    test = load_test()

    test = {(e.eid, e.category): e
            for e in test}

    tdb = list(flatten([load_train(), load_dev()]))

    with open('reg_db.csv', 'r', encoding='utf-8') as f:

        f.readline()

        refs = []

        for i, l in enumerate(f):

            data = l[:-1].split('\t')

            eid, category, ent = data[:3]

            e = test[(eid, category)]

            ref = None

            t = {t for t in e.triples if ent in {t.subject, t.object}}

            if len(e.triples) > 1:

                for e_ in tdb:

                    if len(e_.triples) == 1:

                        if t.intersection(e_.triples):

                            for l in e_.lexes:

                                if l['references']:

                                    for r in l['references']:

                                        if r['entity'] == ent:

                                            ref = r['ref']
                                            break
                    if ref:
                        break
            else:

                for e_ in tdb:

                    if len(e_.triples) > 1:

                        if t.intersection(e_.triples):

                            for l in e_.lexes:

                                if l['references']:

                                    for r in l['references']:

                                        if r['entity'] == ent:

                                            ref = r['ref']
                                            break
                    if ref:
                        break

            if ref is None:
                ref = preprocess_so(data[-2].lower())

            refs.append(ref)

    return refs
