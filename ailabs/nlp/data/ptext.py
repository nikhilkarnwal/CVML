import itertools
import json
import re
from collections import Counter
import numpy as np
from tqdm import tqdm


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def process_punctuation(s):
    # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
    # this version should be faster since we use re instead of repeated operations on str's
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    return s.strip()


def parse(data_path):
    with open(data_path, 'r', encoding='utf-8') as fd:
        lines = fd.read().strip().split('\n')

    pairs = [line.split('\t')[:2] for line in lines]
    lang1_token = []
    lang2_token = []
    for sentence in tqdm(pairs, desc='Processing Tokens'):
        lang1_token.append(process_punctuation(sentence[0]).lower().split(' '))
        lang2_token.append(process_punctuation(sentence[1]).split(' '))
    return lang1_token, lang2_token


def execute(data_path, vocab_path):
    lang1_token, lang2_token = parse(data_path)
    print(lang2_token[:10])
    vocab_1 = extract_vocab(np.array(lang1_token).reshape(-1), start=2)
    vocab_2 = extract_vocab(np.array(lang2_token).reshape(-1), start=2)
    vocab_1['SOS'] = 0
    vocab_1['EOS'] = 1
    vocab_2['SOS'] = 0
    vocab_2['EOS'] = 1
    with open(vocab_path, 'w', encoding='utf-8') as v_fd:
        json.dump({'lang1': vocab_1, 'lang2': vocab_2}, v_fd, indent=4, ensure_ascii=False)
