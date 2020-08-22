import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)

        def save_output(module, input, output):
            self.l4_output = output

        self.resnet.layer4.register_forward_hook(save_output)

    def forward(self, image):
        self.resnet(image)
        return self.l4_output


class Attention(nn.Module):
    def __init__(self, vis_dim, mid_dim, glimpses, query_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(vis_dim, mid_dim, 1, bias=True)
        self.linear = nn.Linear(query_dim, mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, glimpses, 1, bias=True)

    def forward(self, image, query):
        v = self.conv1(image)
        B, C, H, W = v.shape
        q = self.linear(query)
        q = q.repeat(H, W, 1, 1).permute(2, 3, 0, 1)
        vq = torch.relu(v + q)
        a = self.conv2(vq).reshape(B, -1, H * W)
        a = torch.softmax(a, dim=-1).reshape(B, -1, H, W)
        return a


def apply_attention(attn, img):
    N, C, H, W = img.shape
    img = img.view(N, 1, C, -1)  # N*1*C*(H*W)
    attn = attn.view(N, -1, 1, H * W)  # N*G*1*(H*W)
    attn_img = attn * img  # N*G*C*(H*W)
    attn_img = attn_img.sum(dim=-1)  # N*G*C
    return attn_img.view(N, -1)
