from model.vqanet import VQANet
from model.vision import FeatureExtractor, Attention
from torchsummary import summary

if __name__ == "__main__":
    vnet = FeatureExtractor()
    summary(vnet, (3, 256, 256), device='cpu')
    attn = Attention(1024, 512, 5, 256)
    summary(attn, [(1024, 8, 8), (256,)], device='cpu')
    # net = VQANet(100, 1024, 1024, 1, 2056, 1024, 5, 100)
    # summary(net, [(5,), (), (3, 114, 114)], batch_size=2, device='cpu')
