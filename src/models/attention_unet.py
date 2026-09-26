
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two 3x3 convs, each followed by InstanceNorm2d + ReLU (per Section 8)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Filters the encoder skip-connection features using decoder context."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, kernel_size=1), nn.InstanceNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, kernel_size=1), nn.InstanceNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, kernel_size=1), nn.InstanceNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.relu(g + s)
        attn = self.psi(attn)
        return skip * attn


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        chs = [32, 64, 128, 256]
        bottleneck_ch = 512

        self.enc1 = ConvBlock(in_ch, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[3], bottleneck_ch)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(bottleneck_ch, chs[3], kernel_size=3, padding=1))
        self.att4 = AttentionGate(chs[3], chs[3], chs[3] // 2)
        self.dec4 = ConvBlock(chs[3] * 2, chs[3])

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(chs[3], chs[2], kernel_size=3, padding=1))
        self.att3 = AttentionGate(chs[2], chs[2], chs[2] // 2)
        self.dec3 = ConvBlock(chs[2] * 2, chs[2])

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(chs[2], chs[1], kernel_size=3, padding=1))
        self.att2 = AttentionGate(chs[1], chs[1], chs[1] // 2)
        self.dec2 = ConvBlock(chs[1] * 2, chs[1])

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                  nn.Conv2d(chs[1], chs[0], kernel_size=3, padding=1))
        self.att1 = AttentionGate(chs[0], chs[0], chs[0] // 2)
        self.dec1 = ConvBlock(chs[0] * 2, chs[0])

        self.final = nn.Conv2d(chs[0], out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.up4(b); s4 = self.att4(d4, e4); d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4); s3 = self.att3(d3, e3); d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3); s2 = self.att2(d2, e2); d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2); s1 = self.att1(d1, e1); d1 = self.dec1(torch.cat([d1, s1], dim=1))
        return self.final(d1)
