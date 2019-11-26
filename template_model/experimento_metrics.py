# -*- coding: utf-8 -*-

def get_old_cat_ixs():

    with open('../evaluation/subsets/test/old-cat.txt', 'r') as f:

        old_ids = [int(i[:-1]) for i in f.readlines()]

    return old_ids



def get_references0():

    with open('../evaluation/references/test/all-cat_reference0.lex', 'r', encoding='utf-8') as f:

        texts = [l[:-1] for l in f.readlines()]

    return texts


def delete_n_random_texts(n, seed=123):

    refs = get_references0()

    from random import Random

    r = Random(seed)

    old_ids = get_old_cat_ixs()
    r.shuffle(old_ids)
    to_delete_ix = set(old_ids[:n])

    new_refs = []

    for ix, ref in enumerate(refs):

        if ix in to_delete_ix:

            new_refs.append('')
        else:
            new_refs.append(ref)

    eval_refs(new_refs, f'delete{n}')


def repeat_n_random_texts(n, seed=234):

    refs = get_references0()

    from random import Random

    r = Random(seed)

    old_ids = get_old_cat_ixs()
    r.shuffle(old_ids)
    to_repeat_ix = set(old_ids[:n])

    new_refs = []

    for ix, ref in enumerate(refs):

        if ix in to_repeat_ix:

            new_refs.append(2 * ref)
        else:
            new_refs.append(ref)

    eval_refs(new_refs, f'repeat{n}')


def delete_m_random_words_from_n_texts(n, m, seed_n=345, seed_m=444):

    refs = get_references0()

    from random import Random

    r_n = Random(seed_n)
    r_m = Random(seed_m)

    old_ids = get_old_cat_ixs()
    r_n.shuffle(old_ids)
    to_delete_ix = set(old_ids[:n])

    new_refs = []

    for ix, ref in enumerate(refs):

        if ix in to_delete_ix:

            ref_words = ref.split()

            wixs = list(range(len(ref_words)))
            wixs_to_delete = r_m.sample(wixs, min([len(wixs), m]))

            new_ref = ' '.join(w for i, w in enumerate(ref_words) if i not in wixs_to_delete)

            new_refs.append(new_ref)
        else:
            new_refs.append(ref)

    eval_refs(new_refs, f'delete_{m}_from_{n}')


def shuffle_n_texts(n, seed=456, seed_shuffle=567):

    refs = get_references0()

    from random import Random

    r = Random(seed)
    r_shuffle = Random(seed_shuffle)

    old_ids = get_old_cat_ixs()
    r.shuffle(old_ids)
    to_shuffle_ix = set(old_ids[:n])

    new_refs = []

    for ix, ref in enumerate(refs):

        if ix in to_shuffle_ix:

            ref_words = ref.split()
            r_shuffle.shuffle(ref_words)

            shuffled = ' '.join(ref_words)

            new_refs.append(shuffled)
        else:
            new_refs.append(ref)

    eval_refs(new_refs, f'shuffle{n}')


def eval_refs(refs, name):

    import os
    import sys
    sys.path.append('../evaluation')
    from evaluate import preprocess_model_to_evaluate, evaluate_system

    if not os.path.isdir(f'../data/models/test/{name}'):
        os.mkdir(f'../data/models/test/{name}')

    with open(f'../data/models/test/{name}/{name}.txt', 'w', encoding='utf-8') as f:

        for ref in refs:

            f.write(f'{ref}\n')

    preprocess_model_to_evaluate(f'../data/models/test/{name}/{name}.txt', 'test')

    evaluate_system(name, 'test', ['old-cat'])

