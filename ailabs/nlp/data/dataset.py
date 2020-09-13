import json

import numpy
import torch
import torch.utils.data as data
from tqdm import tqdm

from . import ptext


class MLTDataSet(data.Dataset):

    def __init__(self, data_path, vocab_path) -> None:
        super().__init__()
        self.lang1, self.lang2 = ptext.parse(data_path)
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as fd:
            vocab = json.load(fd)
        self.input = [[vocab['lang1']['SOS'],
                       *[vocab['lang1'][token] for token in line],
                       vocab['lang1']['EOS']] for line in tqdm(self.lang1, desc='Converting Text to Index for input')]
        # not adding SOS in target as it does not need to be predicted
        self.target = [[*[vocab['lang2'][token] for token in line],
                        vocab['lang2']['EOS']]
                       for line in tqdm(self.lang2, desc='Converting Text to Index for target')]
        self.start_id = vocab['lang1']['SOS']
        self.end_id = vocab['lang1']['EOS']
        self.vocab = vocab
        self.index_to_token_target = {item[1]: item[0] for item in vocab['lang2'].items()}
        print('Vocab Size-{}'.format(self.get_input_size()))

    def __getitem__(self, index: int):
        return index, self.input[index], self.target[index]

    def __len__(self) -> int:
        return len(self.input)

    def get_start_end(self):
        return self.start_id, self.end_id

    def get_input_size(self):
        return len(self.vocab['lang1']), len(self.vocab['lang2'])

    def save_evaluated(self, answ, indx, loss, eval_filename):
        with open(eval_filename, 'w', encoding='utf-8') as fd:
            pred = []
            for itr, ind in enumerate(indx):
                pred.append({
                    'input': ' '.join(self.lang1[ind]),
                    'target': ' '.join(self.lang2[ind]),
                    'predicted': ' '.join([self.index_to_token_target[i] for i in answ[itr]])
                })
            pred.append({
                'loss': loss
            })
            json.dump(pred, fd, indent=4, ensure_ascii=False)
