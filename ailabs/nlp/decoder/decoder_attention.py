import random

import torch
from torch import nn


class DecoderAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_prob, num_layers, batch_first=True):
        super(DecoderAttentionLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.fc_encoder = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fc_hidden = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.fc_combine = nn.Parameter(torch.FloatTensor(size=(1, hidden_dim)))
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(input_size=2 * hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False,
                            dropout=drop_prob)
        self.classifier = nn.Linear(hidden_dim, input_dim)

    def step_forward(self, _input, _hidden, _encoder_state):
        # embedded = (1,1,H)
        # input -> (1,output_len=1)
        embedded = self.embed(_input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # hidden =(hidden, cell) , so using index-0
        # hidden[0] = (1,H)
        # _encoder_state = (1,Len,H)
        # x = (1,Len,H)
        x = torch.tanh(self.fc_encoder(_encoder_state) + self.fc_hidden(_hidden[0]))
        # a_score = (1,Len,1)
        alignment_score = x.bmm(self.fc_combine.unsqueeze(2))

        # attn_w = (1,Len)
        attn_weight = nn.functional.softmax(alignment_score.view(1, -1), dim=1)

        # context = (1,1,H)
        context_vector = torch.bmm(attn_weight.unsqueeze(0), _encoder_state)

        # output = (1,1,2H)
        output = torch.cat([embedded[0], context_vector[0]], dim=1).unsqueeze(0)

        # output = (1,1,H)
        output, _hidden = self.lstm(output, _hidden)

        # output = (1,Vocab_size/Input_dim) as in output of classifier
        output = nn.functional.log_softmax(self.classifier(output[0]), dim=1)
        return output, _hidden, attn_weight

    def forward(self, start_token, end_token, hidden, target,
                encoder_state, teacher_forcing_prob=0.00):
        # x0 -> (1,len=1)
        x0 = start_token
        # hidden -> (1,1,H)
        # c0 = torch.zeros_like(hidden)
        _hidden = hidden  # (hidden, c0)
        # target -> (OutputLen-OL,batch-1)
        target_len = target.size(0)
        teacher_forcing = True if teacher_forcing_prob > random.random() else False
        output = torch.ones(size=(target_len, self.input_dim), device=target.device)
        for i in range(target_len):
            _output, _hidden, attn_weight = self.step_forward(x0, _hidden, encoder_state)

            if teacher_forcing:
                x0 = target[i].unsqueeze(0)
            else:
                _, x0 = _output.topk(1)
                x0 = x0.detach()
            output[i] = _output.squeeze(0)
            if x0[0].item() == end_token.item():
                break

        return output
