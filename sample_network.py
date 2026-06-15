import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Sequence, Tuple

#org_img = "/CVIP/SATNet/sample_img"
#flipped_img = cv2.flip(org_img, 1) # 좌우반전: f_img = cv2.flip(img, 1)

def align_flipped_feature(x: torch.Tensor)-> torch.Tensor:
    return torch.flip(x, dims=(-1,))
class ECA_fixed(nn.Module):
    def __init__(self, channels: int, gamma: int=2, b: int=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = max(t if t % 2 else t + 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 채널 차원만 연산 -> Conv1d()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)

class concat_comput(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2, x3=None):
        x = torch.cat([x1, x2], dim=1)
        output = self.concat(x)
        
        return output
    
class CCL(nn.Module):
    def __init__(self, channel: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, dilation=dilation), # 3x3
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.dil_conv = nn.Conv2d(channel, channel, 3, dilation=dilation) # 3x3
        self.BN = nn.BatchNorm2d()
        self.relu = nn.ReLU()

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d()

    def forward(self, x):
        f_ct = self.GAP(self.conv1(self.sigmoid(x)))
        f_l = self.relu(self.BN(self.dil_conv(x)))

        output = self.relu(self.BN(f_l - f_ct))

        return output

class CCL_fixed(nn.Module):
    def __init__(self, channel: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.BN = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        f_l = self.conv1(x)
        f_ct = self.conv2(x)

        output = self.relu(self.BN(f_l - f_ct))

        return output

class SAAM(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        
        self.concat_compu = concat_comput(embed_dim * 2, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.eca_feature = ECA_fixed(embed_dim)
        self.eca_flipped = ECA_fixed(embed_dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        x2_aligned = torch.flip(x2, dims=[-1])
        Fc = self.concat_compu(x1, x2_aligned)
        
        q1 = x1.flatten(2).permute(2, 0, 1)
        q2 = x2_aligned.flatten(2).permute(2, 0, 1)
        k = Fc.flatten(2).permute(2, 0, 1)
        v = Fc.flatten(2).permute(2, 0, 1)

        attn_output1, _ = self.mha(q1, k, v)
        attn_output2, _ = self.mha(q2, k, v)

        attn_output1 = attn_output1.permute(1, 2, 0).reshape(B, C, H, W)
        attn_output2 = attn_output2.permute(1, 2, 0).reshape(B, C, H, W)

        F_hat = self.eca_feature(attn_output1)
        F_hat_f = self.eca_flipped(attn_output2)

        return F_hat, F_hat_f

# class CFDM(nn.Module):
#     dilation_dict = {
#         0: 8,
#         1: 6,
#         2: 4, 
#         3: 2,
#     }
#     def __init__(self, channels: int, stage:int, prev_channels:int, num_classes: int=2):
#         super().__init__()
        
#         self.concat_compu = concat_comput()
#         self.CCL = CCL()
#         self.bi_up = nn.UpsamplingBilinear2d(size=None, scale_factor=None)
#             # size: output spatial sizes
#             # scale_factor: multiplier for spatial size.
#         self.act1 = nn.Sigmoid()
#         self.BN = nn.BatchNorm2d()
#         self.conv3 = nn.Conv2d()

#     def forward(self, x1, x2, stage:int):
#         if stage == 3:
#             Fc = self.CCL(self.concat_compu(x1, x2))
#             x1 = self.CCL(x1)
#             x2 = self.CCL(x2)

#             output = self.concat_compu(x1, x2, Fc)

#             # 아직 미완성
#             return output
            
#         else:
#             # 여기선 상위 stage의 Decoder output 같이 연산해줘야 함.
#             D = self.bi_up(self.act1(self.BN(self.conv3("여기다 이전 Stage의 Output"))))

class CFDM(nn.Module):
    """Contrast and Fusion Decoder Module."""

    def __init__(self, channels: int, stage: int, prev_channels: int=None):
        super().__init__()
        dilation_dict = {
        0: 8,
        1: 6,
        2: 4,
        3: 2,
        }

        if stage not in dilation_dict:
            raise ValueError(f"stage must be one of {sorted(self.DILATIONS)}")
        
        dilation = dilation_dict[stage]
        self.stage = stage

        self.concat_compu = concat_comput(channels * 2, channels)
        self.concat_compu_output = concat_comput(channels * 3, channels)

        self.CCL = CCL_fixed(channels, dilation)
        self.bi_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.act1 = nn.ReLU()

        if stage==3:
            self.conv3 = None
            self.BN = None
        else:
            self.conv3 = nn.Conv2d(prev_channels, channels, 3, padding=1)
            self.BN = nn.BatchNorm2d(channels)

    def forward(self, x1, x2, prev_output= None):
        Fc = self.concat_compu(x1, x2)

        if self.stage==3:
            Fc = self.CCL(Fc)
            x1 = self.CCL(x1)
            x2 = self.CCL(x2)

            x12 = torch.cat([x1, x2], dim=1)
            output = self.concat_compu_output(x12, Fc)

            return output
        else:
            if prev_output is None:
                raise ValueError("stage 0, 1, 2 needs prev_output")
            D = self.bi_up(self.act1(self.BN(self.conv3(prev_output))))

            if D.shape[-2:] != x1.shape[-2:]:
                D = torch.nn.functional.interpolate(
                    D, size=x1.shape[-2:], mode = "bilinear", align_corners=False,
                )
            Fc = Fc + D
            x1 = x1 + D
            x2 = x2 + D

            Fc = self.CCL(Fc)
            x1 = self.CCL(x1)
            x2 = self.CCL(x2)

            x12 = torch.cat([x1, x2], dim=1)
            output = self.concat_compu_output(x12, Fc)

            return output

class SATNetDecoder(nn.Module):
    """SATNet decoder/head.

    This class expects four backbone feature maps:
    features = [F0, F1, F2, F3]
    flipped_features = [F0_f, F1_f, F2_f, F3_f]

    The backbone itself is intentionally not included here.
    """

    def __init__(
        self,
        channels: Sequence[int] = (96, 192, 384, 768),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        num_classes: int = 2,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain 4 stage channel sizes.")
        if len(num_heads) != 4:
            raise ValueError("num_heads must contain 4 stage head counts.")

        self.saam2 = SAAM(channels=channels[2], num_heads=num_heads[2])
        self.saam3 = SAAM(channels=channels[3], num_heads=num_heads[3])

        self.cfdm3 = CFDM(
            channels=channels[3],
            stage=3,
            prev_channels=None,
            num_classes=num_classes,
        )
        self.cfdm2 = CFDM(
            channels=channels[2],
            stage=2,
            prev_channels=channels[3],
            num_classes=num_classes,
        )
        self.cfdm1 = CFDM(
            channels=channels[1],
            stage=1,
            prev_channels=channels[2],
            num_classes=num_classes,
        )
        self.cfdm0 = CFDM(
            channels=channels[0],
            stage=0,
            prev_channels=channels[1],
            num_classes=num_classes,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        flipped_features: Sequence[torch.Tensor],
        flipped_features_are_aligned: bool = False,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if len(features) != 4 or len(flipped_features) != 4:
            raise ValueError("SATNetDecoder expects 4 feature maps per path.")

        features = list(features)
        flipped_features = list(flipped_features)

        if not flipped_features_are_aligned:
            flipped_features = [align_flipped_feature(x) for x in flipped_features]

        features[2], flipped_features[2] = self.saam2(
            features[2],
            flipped_features[2],
        )
        features[3], flipped_features[3] = self.saam3(
            features[3],
            flipped_features[3],
        )

        out3, pred3 = self.cfdm3(features[3], flipped_features[3])
        out2, pred2 = self.cfdm2(features[2], flipped_features[2], prev_out=out3)
        out1, pred1 = self.cfdm1(features[1], flipped_features[1], prev_out=out2)
        out0, pred0 = self.cfdm0(features[0], flipped_features[0], prev_out=out1)

        return {
            "out": out0,
            "final_pred": pred0,
            "preds": [pred0, pred1, pred2, pred3],
            "decoder_features": [out0, out1, out2, out3],
        }


if __name__ == "__main__":
    batch_size = 1
    channels = (96, 192, 384, 768)

    features = [
        torch.randn(batch_size, channels[0], 64, 64),
        torch.randn(batch_size, channels[1], 32, 32),
        torch.randn(batch_size, channels[2], 16, 16),
        torch.randn(batch_size, channels[3], 8, 8),
    ]
    flipped_features = [
        torch.randn(batch_size, channels[0], 64, 64),
        torch.randn(batch_size, channels[1], 32, 32),
        torch.randn(batch_size, channels[2], 16, 16),
        torch.randn(batch_size, channels[3], 8, 8),
    ]

    model = SATNetDecoder(channels=channels)
    outputs = model(features, flipped_features)

    print("final_pred:", outputs["final_pred"].shape)
    for idx, pred in enumerate(outputs["preds"]):
        print(f"P{idx}:", pred.shape)