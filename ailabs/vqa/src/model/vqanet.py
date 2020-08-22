import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .vision import FeatureExtractor, Attention, apply_attention
from .text import TextNet


class VQANet(nn.Module):
    def __init__(self, words_dim, vec_dim, lstm_dim, lstm_layers,
                 vis_dim, atten_mid_dim, glimpses, classes):
        super().__init__()

        self.text_net = TextNet(
            words_dim=words_dim,
            vec_dim=vec_dim,
            hid_dim=lstm_dim,
            num_layers=lstm_layers)

        self.vision = FeatureExtractor()

        self.attention = Attention(
            vis_dim=vis_dim,
            mid_dim=atten_mid_dim,
            query_dim=vec_dim,
            glimpses=glimpses
        )

        self.classifier = Classifier(
            in_features=glimpses * vis_dim + vec_dim,
            mid_features=1024,
            out_features=classes,
            drop=0.5
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query, q_len, image):
        word_vec = self.text_net(query, q_len, True)
        vis_feat = self.vision(image)
        vis_feat = vis_feat / (vis_feat.norm(p=2, dim=1, keepdim=True).expand_as(vis_feat) + 1e-8)
        attn = self.attention(word_vec, vis_feat)
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
