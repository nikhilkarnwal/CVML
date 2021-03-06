import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .vision import FeatureExtractor, Attention, apply_attention
from .text import TextNet


class VQANet(nn.Module):
    def __init__(self, words_dim, vec_dim, lstm_dim, lstm_layers,
                 vis_dim, atten_mid_dim, glimpses, classes):
        """
        VQANet is composite neural net consisting of following networks -
        TextNet:
        Attention:
        Classifier:

        :param words_dim: size of vocab
        :type words_dim: int
        :param vec_dim: size of word embeddings
        :type vec_dim: int
        :param lstm_dim: number of hidden units in lstm
        :type lstm_dim: int
        :param lstm_layers: number of lstm layers in lstm net
        :type lstm_layers: int
        :param vis_dim: number of channels in input image feature map
        :type vis_dim: int
        :param atten_mid_dim: number of channels in mid of attention layer
        :type atten_mid_dim: int
        :param glimpses: number of channels in output of attention layer
        :type glimpses: int
        :param classes: number of classes to classify into
        :type classes: int
        """
        super().__init__()

        self.text_net = TextNet(
            words_dim=words_dim,
            vec_dim=vec_dim,
            hid_dim=lstm_dim,
            num_layers=lstm_layers)

        self.attention = Attention(
            vis_dim=vis_dim,
            mid_dim=atten_mid_dim,
            query_dim=lstm_dim,
            glimpses=glimpses
        )

        self.classifier = Classifier(
            in_features=glimpses * vis_dim + lstm_dim,
            mid_features=1024,
            out_features=classes,
            drop=0.5
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query, q_len, vis_feat):
        word_vec = self.text_net(query, q_len, True).squeeze(0)
        vis_feat = vis_feat / (vis_feat.norm(p=2, dim=1, keepdim=True).expand_as(vis_feat) + 1e-8)
        attn = self.attention(vis_feat, word_vec)
        attn_vis = apply_attention(attn, vis_feat)
        combined = torch.cat([attn_vis, word_vec], dim=1)
        answer = self.classifier(combined)
        return answer


class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))
