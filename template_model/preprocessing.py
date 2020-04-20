# -*- coding: utf-8 -*-
import re
from unidecode import unidecode


#RE_SPACE_AFTER_COMMA = re.compile(r',(\w)')
#RE_SPACES_INSIDE_PARENS = re.compile(r'\(\s?(.*?)\s?\)')
#RE_SPACE_BEFORE_PARENS = re.compile(r'(\S)\(')
#RE_WEIRD_QUOTES_MARKS2 = re.compile(r'`` (.*?) \'\'')
#RE_WEIRD_QUOTES_MARKS3 = re.compile(r'\'\' (.*?) \'\'')
#RE_WEIRD_QUOTES_MARKS4 = re.compile(r'\' (.*?) \'')
#RE_WEIRD_QUOTES_MARKS5 = re.compile(r'` (.*?) \'')
#RE_WEIRD_MINUS_SIGNS = re.compile(r'\s--\s')
RE_SPACE_BEFORE_COMMA_DOT = re.compile(r'\s(?=\.|,|:|;|!)')
RE_WEIRD_QUOTE_MARKS = re.compile(r'(`{1,2})(?!s)')
RE_APOSTROPH_S = re.compile(r'\s[\"\']s')
RE_SPACES_INSIDE_QUOTES = re.compile(r'(?<=")(\s(.*?)\s)(?=")')

def normalize_thiagos_template(s):

    s = RE_SPACE_BEFORE_COMMA_DOT.sub('', s)
    s = RE_WEIRD_QUOTE_MARKS.sub('"', s)
    s = RE_APOSTROPH_S.sub('\'s', s)

    return RE_SPACES_INSIDE_QUOTES.sub(r'\g<2>', s)


def text_to_id(text):

    return text.replace(' ', '_')


TOKENIZER_RE = re.compile(r'(\W)')
def normalize_text(text):

    lex_detokenised = ' '.join(TOKENIZER_RE.split(text))
    lex_detokenised = ' '.join(lex_detokenised.split())

    return unidecode(lex_detokenised.lower())


RE_SPLIT_DOT_COMMA = re.compile(r'([\.,\'])')

def preprocess_text(t):

    return ' '.join(' '.join(RE_SPLIT_DOT_COMMA.split(t)).split())


def preprocess_text2(t):

    tt = (' '.join(' '.join(RE_SPLIT_DOT_COMMA.split(t)).split())).replace('@', '').lower()

    tt = normalize_thiagos_template(tt)

    # se há um ponto final grudado numa palavra, separa ele da palavra que vem antes
    if tt[-1] == '.' and tt[-2] != ' ':
        tt = f'{tt[:-1]} .'
    
    return tt


PARENTHESIS_RE = re.compile(r'(.*?)\((.*?)\)')
CAMELCASE_RE = re.compile(r'([a-z])([A-Z])')

def preprocess_so(so):

    parenthesis_preprocessed = PARENTHESIS_RE.sub(r'\g<2> \g<1>', so)
    underline_removed = parenthesis_preprocessed.replace('_', ' ')
    camelcase_preprocessed = CAMELCASE_RE.sub(r'\g<1> \g<2>',
                                              underline_removed)

    return camelcase_preprocessed.strip('" ')
