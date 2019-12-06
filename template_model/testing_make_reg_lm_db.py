# -*- coding: utf-8 -*-
import re
from collections import Counter


RE_MATCH_TEMPLATE_KEYS_LEX = re.compile((r'(AGENT\\-\d|PATIENT\\-\d'
                                         r'|BRIDGE\\-\d)'))
RE_MATCH_TEMPLATE_KEYS = re.compile(r'(AGENT\-\d|PATIENT\-\d|BRIDGE\-\d)')
TRANS_ESCAPE_TO_RE = str.maketrans('-', '_', '\\')


def extract_text_reg_lm(l):

    s = l['text']
    t = l['normalized_template']

    # permite capturar entidades que aparecem mais de uma vez no template
    # ex:
    #   AGENT-1 comeu PATIENT-1 e AGENT-1 vai trabalhar.
    #   João comeu carne e ele vai trabalhar.
    # como uso um regex com grupos nomeados de acordo com o label da entidade
    #    e só posso ter um grupo por label, é preciso criar labels diferentes
    #    para AGENT-1
    # para tanto, adiciono um contado à frente do label, ficando:
    # (?P<AGENT_1_1>.*?) comeu (?P<PATIENT_1_1>.*?) e (?P<AGENT_1_2>.*?)
    # vai trabalhar.

    # contador, por label, de quantos grupos regex já foram utilizados
    lex_counts = Counter()

    def replace_sop_to_catch(m):

        entity = m.group(0)
        lex_counts[entity] += 1

        # cria o nome do grupo a partir do label
        # como a string está escapada, os labels são recebidos como, ex:
        #    AGENT\\-1
        # devendo ser transformado, para ser utilizado como grupo em:
        #    AGENT_1_1
        # o que ocorre em dois passos:
        #    .translate(TRANS_ESCAPE_TO_RE) - remove \ e troca - por _
        #    '{}_{}'.format(_, lex_counts[entity]) - add o contador de entidade
        entity_group = '{}_{}'.format(entity.translate(TRANS_ESCAPE_TO_RE),
                                      lex_counts[entity])

        # retorna então o regex do grupo que irá capturar
        #    os caracteres da entidade
        return '(?P<{v}>.*?)'.format(v=entity_group)

    lex_counts = Counter()

    def replace_sop_to_fill(m):

        entity = m.group(0)
        lex_counts[entity] += 1

        entity_group = '{}_{}'.format(entity.translate(TRANS_ESCAPE_TO_RE),
                                      lex_counts[entity])

        return '{{{}}}'.format(entity_group)

    # substitui os labels de entidades por regex para capturar suas substrings
    #    adiciona ^ e $ para delimitar o início e fim da string
    t_re = '^{}$'.format(RE_MATCH_TEMPLATE_KEYS_LEX.sub(replace_sop_to_catch,
                                                        re.escape(t)))
    to_fill = RE_MATCH_TEMPLATE_KEYS.sub(replace_sop_to_fill,
                                         t)

    m = re.match(t_re, s)

    if m:

        captureds = m.groupdict()
        as_keys = {k: v.replace(' ', '_') for k, v in captureds.items()}

        return to_fill.format(**as_keys)
    return None
