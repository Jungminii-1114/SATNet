import torch
from sample_network import SAAM, CFDM, ECA_fixed, concat_comput, CCL_fixed

B, C, H, W = 2, 96, 64, 64
x1 = torch.randn(B, C, H, W)
x2 = torch.randn(B, C, H, W)

eca = ECA_fixed(C)
print("ECA:", eca(x1).shape)

fuse = concat_comput(C * 2, C)
print("concat:", fuse(x1, x2).shape)

ccl = CCL_fixed(C, dilation=8)
print("CCL:", ccl(x1).shape)

saam = SAAM(embed_dim=96, num_heads=3)
y1, y2 = saam(x1, x2)
print(y1.shape, y2.shape)

x1_3 = torch.randn(1, 768, 8, 8)
x2_3 = torch.randn(1, 768, 8, 8)

x1_2 = torch.randn(1, 384, 16, 16)
x2_2 = torch.randn(1, 384, 16, 16)

x1_1 = torch.randn(1, 192, 32, 32)
x2_1 = torch.randn(1, 192, 32, 32)

x1_0 = torch.randn(1, 96, 64, 64)
x2_0 = torch.randn(1, 96, 64, 64)

cfdm3 = CFDM(channels=768, stage=3)
out3 = cfdm3(x1_3, x2_3)
print("CFDM3:", out3.shape)

cfdm2 = CFDM(channels=384, stage=2, prev_channels=768)
out2 = cfdm2(x1_2, x2_2, prev_output=out3)
print("CFDM2:", out2.shape)

cfdm1 = CFDM(channels=192, stage=1, prev_channels=384)
out1 = cfdm1(x1_1, x2_1, prev_output=out2)
print("CFDM1:", out1.shape)

cfdm0 = CFDM(channels=96, stage=0, prev_channels=192)
out0 = cfdm0(x1_0, x2_0, prev_output=out1)
print("CFDM0:", out0.shape)