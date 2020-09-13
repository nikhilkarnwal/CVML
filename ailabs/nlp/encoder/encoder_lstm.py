import torch
from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, lstm_layers=1, drop_prob=0, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.drop_prob = drop_prob
        self.embed = nn.Embedding(input_size, hidden_size)
        self.drop = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, dropout=drop_prob, batch_first=batch_first)

    def forward(self, inputs, hidden):
        embedded = self.drop(self.embed(inputs))
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    # return hidden and cell state
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_size),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size))
