import subprocess
import re
import os
from unidecode import unidecode
import glob
from itertools import product
import csv


WEBNLG_USED_REFERENCES = [0, 1, 2]


BLEU_RE = re.compile((r'BLEU\ =\ (?P<bleu>.*?),\ (?P<bleu_1>.*?)/'
                      r'(?P<bleu_2>.*?)/(?P<bleu_3>.*?)/(?P<bleu_4>.*?)\ '))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_BLEU_PATH = os.path.join(BASE_DIR, 'tools/multi-bleu.perl')
METEOR_PATH = os.path.join(BASE_DIR, 'tools/meteor-1.5/meteor-1.5.jar')
METEOR_PARAPHRASE_PATH = os.path.join(BASE_DIR,
                                      'tools/meteor-1.5/data/paraphrase-en.gz')
TER_PATH = os.path.join(BASE_DIR, 'tools/tercom-0.7.25/tercom.7.25.jar')


def model_preprocessed_filepath(model, subset, method=None):

    if method == 'ter':
        return os.path.join(BASE_DIR,
                            f'../data/models/{model}/{model}_{subset}_ter.lex')

    return os.path.join(BASE_DIR,
                        f'../data/models/{model}/{model}_{subset}.lex')


def bleu(model, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, subset)

    bleu_references = [os.path.join(BASE_DIR,
                                    f'references/{subset}_reference{r}.lex')
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


METEOR_RE = re.compile(r'Final score:\s+([\d\.]+)\n')


def meteor(model, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, subset)

    refs_str = '_'.join(str(r) for r in references)
    meteor_references = os.path.join(BASE_DIR,
                                     (f'references/{subset}_reference'
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
                    '3',
                    '-a',
                    METEOR_PARAPHRASE_PATH
             ],
            stdout=subprocess.PIPE)

    return [dict(subset=subset,
                 references=references,
                 metric='meteor',
                 value=METEOR_RE.findall(
                         meteor_result.stdout.decode('utf-8'))[0])]


TER_RE = re.compile(r'Total\ TER:\ ([\d\.]+)\ \(')


def ter(model, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, subset, 'ter')

    refs_str = '_'.join(str(r) for r in references)
    ter_references = os.path.join(BASE_DIR,
                                  (f'references/{subset}_reference'
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


def avg_n_tokens(model, subset, references):

    preprocessed_filepath = model_preprocessed_filepath(model, subset)

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


def avg_n_stop_words(model, subset, references):

    from nltk.corpus import stopwords

    sws = stopwords.words('english')

    preprocessed_filepath = model_preprocessed_filepath(model, subset)

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


def evaluate_all_systems():

    systems_filepaths = glob.glob(os.path.join(BASE_DIR,
                                               '../data/models/*'))

    # yes, I'm using system and model interchangeably
    model_names = [os.path.basename(s) for s in systems_filepaths]

    for model in model_names:

        evaluate_system(model)


def evaluate_system(model, subsets=None, references_list=None, methods=None):

    if subsets is None:
        subsets = [os.path.basename(s).split('.')[0]
                   for s in glob.glob(os.path.join(BASE_DIR,
                                                   'subsets/*.txt'))]

    if references_list is None:
        references_list = [WEBNLG_USED_REFERENCES]

    if methods is None:
        methods = SYSTEM_EVALUATION_METHODS.keys()

    system_eval_filepath = os.path.join(BASE_DIR,
                                        (f'../data/models/{model}/'
                                         'system_evaluation.csv'))

    calculated_evals = get_already_calculated_system_eval(system_eval_filepath)

    all_results = []

    for subset, references in product(subsets, references_list):

        for method in methods:

            if (subset, str(references), method) not in calculated_evals:

                eval_function = SYSTEM_EVALUATION_METHODS[method]
                metrics_values = eval_function(model, subset, references)

                all_results.extend(metrics_values)

    already_created = os.path.isfile(system_eval_filepath)

    with open(system_eval_filepath, 'a', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subset',
                                               'references',
                                               'metric',
                                               'value'])

        if not already_created:
            writer.writeheader()
        writer.writerows(all_results)

    return system_eval_filepath


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


def preprocess_all_models():

    models_files = glob.glob(os.path.join(BASE_DIR,
                                          '../data/models/**/*.txt'))

    subsets_files = glob.glob(os.path.join(BASE_DIR,
                                           'subsets/*.txt'))

    for model_file, subset_file in product(models_files, subsets_files):

        preprocess_to_evaluate(model_file, subset_file)


def make_subset_ids_file(s, name):
    # creates a file containing the eid number of the subset entries
    #   subtracts 1, so it starts from 0 and indicates in which lines
    #   of the test text file the entries are located
    ss = s.edf.eid.str.replace(r'Id(\d+)', r'\1').astype(int)
    ss = ss - 1
    ss.to_csv(name, index=False, header=False)


def make_reference_bleu_files(entries_lexes, subset_file, references):

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
                                f'references/{subset}_reference{i}.lex')
        with open(out_file, 'w', encoding='utf-8') as f:
            for lexe in out_lexes:
                f.write(f'{lexe}\n')


def make_reference_meteor_files(entries_lexes, subset_file, references):

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
                            (f'references/{subset}_reference'
                             f'{refs_descr}.meteor'))
    with open(out_file, 'w', encoding='utf-8') as f:
        for lexe in out_lexes:
            f.write(f'{lexe}\n')


def make_reference_ter_files(entries_lexes, subset_file, references):

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
                            f'references/{subset}_reference{refs_descr}.ter')
    with open(out_file, 'w', encoding='utf-8') as f:
        for lexe in out_lexes:
            f.write(f'{lexe}\n')


def make_reference_files(references=[0, 1, 2]):

    import xml.etree.ElementTree as ET

    tree = ET.parse(os.path.join(BASE_DIR, 'testdata_with_lex.xml'))
    root = tree.getroot()

    entries_lexes = []

    for entry in root.iter('entry'):

        lexes = [e.text for e in entry.findall('lex')]
        preprocessed_lexes = [normalize_text(l) for l in lexes]
        entries_lexes.append(preprocessed_lexes)

    subsets_files = glob.glob(os.path.join(BASE_DIR, 'subsets/*.txt'))

    for subset_file in subsets_files:

        make_reference_bleu_files(entries_lexes, subset_file, references)
        make_reference_meteor_files(entries_lexes, subset_file, references)
        make_reference_ter_files(entries_lexes, subset_file, references)
