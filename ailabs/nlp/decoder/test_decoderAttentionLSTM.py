from unittest import TestCase

import torch

from .decoder_attention import DecoderAttentionLSTM


class TestDecoderAttentionLSTM(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.hidden_dim = 32
        self.encoder_len = 10
        self.batch_size = 1
        self.vocab_dim = 1000
        self.num_layers = 1
        self.target_len = 5

        self.encoder_state = torch.FloatTensor(size=(self.batch_size, self.encoder_len, self.hidden_dim))
        self.input = torch.LongTensor(size=(self.batch_size, 1)) % self.vocab_dim
        self.hidden = torch.FloatTensor(size=(self.num_layers, self.batch_size, self.hidden_dim))
        self.cell = torch.zeros_like(self.hidden)
        self.net = DecoderAttentionLSTM(self.vocab_dim, self.hidden_dim, 0, self.num_layers, False)
        self.net.eval()
        self.target = torch.LongTensor(self.target_len, self.batch_size)

    def test_step_forward(self):
        output, _hidden, attn_wght = self.net.step_forward(self.input, _hidden=(self.hidden, self.cell),
                                                           _encoder_state=self.encoder_state)
        self.assertEqual(torch.Size([self.batch_size, self.vocab_dim]), output.size())
        self.assertEqual(torch.Size([self.batch_size, self.encoder_len]), attn_wght.size())

    def test_forward(self):
        output = self.net.forward(torch.LongTensor([0]), torch.LongTensor([1]), (self.hidden, self.cell), self.target,
                                  self.encoder_state, 0.0)
        self.assertEqual(torch.Size([self.target_len, self.vocab_dim]), output.size())
