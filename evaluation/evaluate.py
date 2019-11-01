import subprocess
import re
import os
from unidecode import unidecode
import glob
from itertools import product
import csv
from decimal import Decimal
import sys
sys.path.append('../template_model')
from reading_thiagos_templates import load_shared_task_test, load_dev


WEBNLG_USED_REFERENCES = [0, 1, 2]

BLEU_RE = re.compile((r'BLEU\ =\ (?P<bleu>.*?),\ (?P<bleu_1>.*?)/'
                      r'(?P<bleu_2>.*?)/(?P<bleu_3>.*?)/(?P<bleu_4>.*?)\ '))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_BLEU_PATH = os.path.join(BASE_DIR, 'tools/multi-bleu.perl')
METEOR_PATH = os.path.join(BASE_DIR, 'tools/meteor-1.5/meteor-1.5.jar')
METEOR_PARAPHRASE_PATH = os.path.join(BASE_DIR,
                                      'tools/meteor-1.5/data/paraphrase-en.gz')
TER_PATH = os.path.join(BASE_DIR, 'tools/tercom-0.7.25/tercom.7.25.jar')


def model_preprocessed_filepath(model, set_, subset, method=None):

    if method == 'ter':
        filepath = os.path.join(BASE_DIR,
                                (f'../data/models/{set_}/{model}/'
                                 f'{model}_{subset}_ter.lex'))
    else:
        filepath = os.path.join(BASE_DIR,
                                (f'../data/models/{set_}/{model}/'
                                 f'{model}_{subset}.lex'))

    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    return filepath


def bleu(model, set_, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, set_, subset)

    bleu_references = [os.path.join(BASE_DIR,
                                    (f'references/{set_}/'
                                     f'{subset}_reference{r}.lex'))
                       for r in references]

    with open(preprocessed_filepath, 'rb') as f:
        bleu_result = subprocess.run(
                [MULTI_BLEU_PATH, '-lc'] + bleu_references,
                stdout=subprocess.PIPE,
                input=f.read()
                        )

    results = []

    for k, v in BLEU_RE.match(
            bleu_result.stdout.decode('utf-8')).groupdict().items():

        results.append(dict(subset=subset,
                            references=references,
                            metric=k,
                            value=v))

    return results


METEOR_SYSTEM_RE = re.compile(r'Final score:\s+([\d\.]+)')
METEOR_SENTENCE_RE = re.compile(r'Segment.*?score:\s+(.*?)\n')


def meteor(model, set_, subset, references, level='system'):

    preprocessed_filepath = model_preprocessed_filepath(model, set_, subset)

    refs_str = '_'.join(str(r) for r in references)
    meteor_references = os.path.join(BASE_DIR,
                                     (f'references/{set_}/{subset}_reference'
                                      f'{refs_str}.meteor'))

    meteor_result = subprocess.run(
            [
                    'java',
                    '-Xmx2G',
                    '-jar',
                    METEOR_PATH,
                    preprocessed_filepath,
                    meteor_references,
                    '-l',
                    'en',
                    '-norm',
                    '-r',
                    str(len(references)),
                    '-a',
                    METEOR_PARAPHRASE_PATH
             ],
            stdout=subprocess.PIPE).stdout.decode('utf-8')

    if level == 'system':

        return [dict(subset=subset,
                     references=references,
                     metric='meteor',
                     value=METEOR_SYSTEM_RE.findall(
                             meteor_result)[0])]
    elif level == 'sentence':

        return list(map(Decimal, METEOR_SENTENCE_RE.findall(meteor_result)))


TER_RE = re.compile(r'Total\ TER:\ ([\d\.]+)\ \(')


def ter(model, set_, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model,
                                                        set_, subset, 'ter')

    refs_str = '_'.join(str(r) for r in references)
    ter_references = os.path.join(BASE_DIR,
                                  (f'references/{set_}/{subset}_reference'
                                   f'{refs_str}.ter'))

    ter_result = subprocess.run(
            [
                    'java',
                    '-jar',
                    TER_PATH,
                    '-r',
                    ter_references,
                    '-h',
                    preprocessed_filepath
            ],
            stdout=subprocess.PIPE)

    return [dict(subset=subset,
                 references=references,
                 metric='ter',
                 value=TER_RE.findall(ter_result.stdout.decode('utf-8'))[0])]


def avg_n_tokens(model, set_, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, set_, subset)

    total_tokens = 0
    total_texts = 0

    with open(preprocessed_filepath, 'r', encoding='utf-8') as f:

        for line in f:

            total_texts += 1
            total_tokens += len(line.split(' '))

    return [dict(subset=subset,
                 references=references,
                 metric='avg_n_tokens',
                 value=total_tokens / total_texts)]


def avg_n_stop_words(model, set_, subset, references):

    from nltk.corpus import stopwords

    sws = stopwords.words('english')

    preprocessed_filepath = model_preprocessed_filepath(model, set_, subset)

    total_stops = 0
    total_tokens = 0
    avg_n_stop_word = []

    with open(preprocessed_filepath, 'r', encoding='utf-8') as f:

        for line in f:

            tokens = line.split(' ')
            stops = [w for w in tokens if w in sws]
            total_stops += len(stops)
            total_tokens += len(tokens)
            avg_n_stop_word.append(len(stops) / len(tokens))

    return [dict(subset=subset,
                 references=references,
                 metric='avg_n_stop_words',
                 value=total_stops / total_tokens),
            dict(subset=subset,
                 references=references,
                 metric='macro_avg_n_stop_words',
                 value=sum(avg_n_stop_word) / len(avg_n_stop_word))]


SYSTEM_EVALUATION_METHODS = {'bleu': bleu,
                             'meteor': meteor,
                             'ter': ter,
                             'avg_n_tokens': avg_n_tokens,
                             'avg_n_stop_words': avg_n_stop_words}


def get_already_calculated_system_eval(system_eval_filepath):

    if not os.path.isfile(system_eval_filepath):
        return []

    with open(system_eval_filepath, 'r', encoding='utf-8') as f:

        reader = csv.DictReader(f)

        calculated_evals = [(r['subset'], r['references'], r['metric'])
                            for r in reader]

        return calculated_evals


def evaluate_all_systems(set_,
                         subsets=None, references_list=None, methods=None):

    systems_filepaths = glob.glob(os.path.join(BASE_DIR,
                                               f'../data/models/{set_}/*'))

    # yes, I'm using system and model interchangeably
    model_names = [os.path.basename(s) for s in systems_filepaths]

    for model in model_names:

        print(model)

        evaluate_system(model, set_, subsets, references_list, methods)


def evaluate_system(model,
                    set_, subsets=None, references_list=None, methods=None):

    if subsets is None:
        subsets = [os.path.basename(s).split('.')[0]
                   for s in glob.glob(os.path.join(BASE_DIR,
                                                   f'subsets/{set_}/*.txt'))]

    if references_list is None:
        references_list = [WEBNLG_USED_REFERENCES]

    if methods is None:
        methods = SYSTEM_EVALUATION_METHODS.keys()

    system_eval_filepath = os.path.join(BASE_DIR,
                                        (f'../data/models/{set_}/{model}/'
                                         'system_evaluation.csv'))

    calculated_evals = get_already_calculated_system_eval(system_eval_filepath)

    all_results = []

    for subset, references in product(subsets, references_list):

        for method in methods:

            if (subset, str(references), method) not in calculated_evals:

                eval_function = SYSTEM_EVALUATION_METHODS[method]
                try:
                    metrics_values = eval_function(model,
                                                   set_, subset, references)

                    all_results.extend(metrics_values)
                except FileNotFoundError:
                    pass

    already_created = os.path.isfile(system_eval_filepath)

    with open(system_eval_filepath, 'a', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subset',
                                               'references',
                                               'metric',
                                               'value'])

        if not already_created:
            writer.writeheader()
        writer.writerows(all_results)

    return all_results


TOKENIZER_RE = re.compile(r'(\W)')


def normalize_text(text):

    lex_detokenised = ' '.join(TOKENIZER_RE.split(text))
    lex_detokenised = ' '.join(lex_detokenised.split())

    return unidecode(lex_detokenised.lower())


def preprocess_to_evaluate(model_file, subset_filepath):

    filename = os.path.basename(model_file).split('.')[0]
    subset = os.path.basename(subset_filepath).split('.')[0]
    filedir = os.path.dirname(os.path.abspath(model_file))
    g_file = os.path.join(
                filedir,
                '{filename}_{subset}.lex'.format(filename=filename,
                                                 subset=subset)
            )
    ter_file = os.path.join(
                filedir,
                '{filename}_{subset}_ter.lex'.format(filename=filename,
                                                     subset=subset)
            )

    with open(subset_filepath, 'r', encoding='utf-8') as f:
        lines_to_retain = {int(l) for l in f.readlines()}

    with open(model_file, 'r', encoding='utf-8') as f,\
            open(g_file, 'w', encoding='utf-8') as f_g,\
            open(ter_file, 'w', encoding='utf-8') as f_ter:

        current_out_line = 1

        for line, text in enumerate(f.readlines()):

            if line in lines_to_retain:

                preprocessed_text = normalize_text(text)

                f_g.write(f'{preprocessed_text}\n')
                f_ter.write(f'{preprocessed_text} (id{current_out_line})\n')

                current_out_line += 1


def preprocess_model_to_evaluate(model_file, set_):

    subsets_files = glob.glob(os.path.join(BASE_DIR,
                                           f'subsets/{set_}/*.txt'))

    for subset_file in subsets_files:

        preprocess_to_evaluate(model_file, subset_file)


def preprocess_all_models(set_):

    models_files = glob.glob(os.path.join(BASE_DIR,
                                          f'../data/models/{set_}/**/*.txt'))

    for model_file in models_files:

        preprocess_model_to_evaluate(model_file, set_)


def make_reference_bleu_files(entries_lexes, set_, subset_file, references):

    subset = os.path.basename(subset_file).split('.')[0]

    with open(subset_file, 'r', encoding='utf-8') as f:
        lines_to_retain = {int(l) for l in f.readlines()}

    # make bleu
    for i in references:

        out_lexes = []

        for line, lexes in enumerate(entries_lexes):

            if line in lines_to_retain:

                if len(lexes) > i:
                    out_lexes.append(lexes[i])
                else:
                    out_lexes.append('')

        out_file = os.path.join(BASE_DIR,
                                f'references/{set_}/{subset}_reference{i}.lex')
        with open(out_file, 'w', encoding='utf-8') as f:
            for lexe in out_lexes:
                f.write(f'{lexe}\n')


def make_reference_meteor_files(entries_lexes, set_, subset_file, references):

    subset = os.path.basename(subset_file).split('.')[0]

    with open(subset_file, 'r', encoding='utf-8') as f:
        lines_to_retain = {int(l) for l in f.readlines()}

    out_lexes = []
    refs_descr = '_'.join(str(r) for r in references)

    for line, lexes in enumerate(entries_lexes):

        if line in lines_to_retain:

            for reference in references:

                if len(lexes) > reference:

                    out_lexes.append(lexes[reference])
                else:
                    out_lexes.append('')

    out_file = os.path.join(BASE_DIR,
                            (f'references/{set_}/{subset}_reference'
                             f'{refs_descr}.meteor'))
    with open(out_file, 'w', encoding='utf-8') as f:
        for lexe in out_lexes:
            f.write(f'{lexe}\n')


def make_reference_ter_files(entries_lexes, set_, subset_file, references):

    subset = os.path.basename(subset_file).split('.')[0]

    with open(subset_file, 'r', encoding='utf-8') as f:
        lines_to_retain = {int(l) for l in f.readlines()}

    out_lexes = []
    refs_descr = '_'.join(str(r) for r in references)
    current_out_line = 1

    for line, lexes in enumerate(entries_lexes):

        if line in lines_to_retain:

            for reference in references:

                if len(lexes) > reference:

                    out_lexes.append('{} (id{})'.format(lexes[reference],
                                                        current_out_line))

            current_out_line += 1

    out_file = os.path.join(BASE_DIR,
                            (f'references/{set_}/'
                             f'{subset}_reference{refs_descr}.ter'))
    with open(out_file, 'w', encoding='utf-8') as f:
        for lexe in out_lexes:
            f.write(f'{lexe}\n')


def make_reference_files(set_, references=[0, 1, 2]):

    if set_ == 'test':
        entries = load_shared_task_test()
    elif set_ == 'dev':
        entries = load_dev()

    entries_lexes = [[normalize_text(l['text']) for l in e.lexes]
                     for e in entries]

    subsets_files = glob.glob(os.path.join(BASE_DIR, f'subsets/{set_}/*.txt'))

    for subset_file in subsets_files:

        make_reference_bleu_files(entries_lexes, set_, subset_file, references)
        make_reference_meteor_files(entries_lexes, set_, subset_file, references)
        make_reference_ter_files(entries_lexes, set_, subset_file, references)
