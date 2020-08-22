import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextNet(nn.Module):
    def __init__(self, words_dim, vec_dim,
                 hid_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(words_dim, vec_dim)
        self.lstm = nn.LSTM(input_size=vec_dim,
                            hidden_size=hid_dim,
                            num_layers=num_layers)

    def forward(self, query, query_lens, batch_first=True):
        embedding = torch.tanh(self.embed(query))
        if query_lens is not None:
            packed_seq = pack_padded_sequence(embedding, query_lens, batch_first=batch_first)
        else:
            packed_seq = embedding
        _, (h, _) = self.lstm(packed_seq, batch_first=batch_first)
        return h
