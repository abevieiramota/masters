import subprocess
import re
import os
from unidecode import unidecode
import glob
from itertools import product


BLEU_RE = re.compile(r'BLEU\ =\ ([\d\.]*),')
METEOR_RE = re.compile(r'Final score:\s+([\d\.]+)\n')
TER_RE = re.compile(r'Total\ TER:\ ([\d\.]+)\ \(')


def evaluate_texts(model, subset, references):

    result = {}

    preprocessed_filepath = f'../data/models/{model}/{model}_{subset}.lex'
    preprocessed_ter_filepath = (f'../data/models/{model}/{model}_{subset}'
                                 '_ter.lex')

    # bleu
    bleu_references = [f'references/{subset}_reference{r}.lex'
                       for r in references]

    with open(preprocessed_filepath, 'rb') as f:
        bleu_result = subprocess.run(
                ['tools/multi-bleu.perl', '-lc'] + bleu_references,
                stdout=subprocess.PIPE,
                input=f.read()
                        )

    result['bleu'] = float(BLEU_RE.findall(
            bleu_result.stdout.decode('utf-8'))[0])

    # meteor
    refs_str = '_'.join(str(r) for r in references)

    meteor_references = f'references/{subset}_reference{refs_str}.meteor'

    meteor_result = subprocess.run(
            [
                    'java',
                    '-Xmx2G',
                    '-jar',
                    'tools/meteor-1.5/meteor-1.5.jar',
                    preprocessed_filepath,
                    meteor_references,
                    '-l',
                    'en',
                    '-norm',
                    '-r',
                    '3',
                    '-a',
                    'tools/meteor-1.5/data/paraphrase-en.gz'
             ],
            stdout=subprocess.PIPE)

    result['meteor'] = float(METEOR_RE.findall(
            meteor_result.stdout.decode('utf-8'))[0])

    # ter
    ter_references = f'references/{subset}_reference{refs_str}.ter'

    ter_result = subprocess.run(
            [
                    'java',
                    '-jar',
                    'tools/tercom-0.7.25/tercom.7.25.jar',
                    '-r',
                    ter_references,
                    '-h',
                    preprocessed_ter_filepath
            ],
            stdout=subprocess.PIPE)

    result['ter'] = float(TER_RE.findall(ter_result.stdout.decode('utf-8'))[0])

    return result


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

    with open(f'subsets/{subset}.txt', 'r', encoding='utf-8') as f:
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

    models_files = glob.glob('../data/models/**/*.txt')

    subsets_files = glob.glob('subsets/*.txt')

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

        with open(f'references/{subset}_reference{i}.lex',
                  'w', encoding='utf-8') as f:
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

    with open(f'references/{subset}_reference{refs_descr}.meteor',
              'w', encoding='utf-8') as f:
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

    with open(f'references/{subset}_reference{refs_descr}.ter',
              'w', encoding='utf-8') as f:
        for lexe in out_lexes:
            f.write(f'{lexe}\n')


def make_reference_files(references=[0, 1, 2]):

    import xml.etree.ElementTree as ET

    tree = ET.parse('testdata_with_lex.xml')
    root = tree.getroot()

    entries_lexes = []

    for entry in root.iter('entry'):

        lexes = [e.text for e in entry.findall('lex')]
        preprocessed_lexes = [normalize_text(l) for l in lexes]
        entries_lexes.append(preprocessed_lexes)

    subsets_files = glob.glob('subsets/*.txt')

    for subset_file in subsets_files:

        make_reference_bleu_files(entries_lexes, subset_file, references)
        make_reference_meteor_files(entries_lexes, subset_file, references)
        make_reference_ter_files(entries_lexes, subset_file, references)
