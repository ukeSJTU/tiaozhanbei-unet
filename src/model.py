import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SegmentationUNet(nn.Module):
    """
    UNet specifically designed for multi-class semantic segmentation.
    Optimized for Gear dataset with multiple defect classes.
    """
    def __init__(self, n_channels=3, n_classes=4, bilinear=False, dropout=0.1):
        super(SegmentationUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dropout to bottleneck
        x5 = self.dropout(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class AnomalyUNet(nn.Module):
    """
    UNet specifically designed for anomaly detection.
    Outputs both reconstruction and anomaly segmentation.
    """
    def __init__(self, n_channels=3, bilinear=False):
        super(AnomalyUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Shared encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Reconstruction decoder
        self.up1_recon = Up(1024, 512 // factor, bilinear)
        self.up2_recon = Up(512, 256 // factor, bilinear)
        self.up3_recon = Up(256, 128 // factor, bilinear)
        self.up4_recon = Up(128, 64, bilinear)
        self.outc_recon = OutConv(64, n_channels)
        
        # Anomaly segmentation decoder
        self.up1_seg = Up(1024, 512 // factor, bilinear)
        self.up2_seg = Up(512, 256 // factor, bilinear)
        self.up3_seg = Up(256, 128 // factor, bilinear)
        self.up4_seg = Up(128, 64, bilinear)
        self.outc_seg = OutConv(64, 1)

    def forward(self, x):
        # Shared encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Reconstruction branch
        recon = self.up1_recon(x5, x4)
        recon = self.up2_recon(recon, x3)
        recon = self.up3_recon(recon, x2)
        recon = self.up4_recon(recon, x1)
        reconstruction = torch.sigmoid(self.outc_recon(recon))
        
        # Anomaly segmentation branch
        seg = self.up1_seg(x5, x4)
        seg = self.up2_seg(seg, x3)
        seg = self.up3_seg(seg, x2)
        seg = self.up4_seg(seg, x1)
        anomaly_map = torch.sigmoid(self.outc_seg(seg))
        
        return reconstruction, anomaly_map


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test standard UNet
    model = UNet(n_channels=3, n_classes=1)
    model = model.to(device)
    
    # Test input
    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"UNet input shape: {x.shape}")
    print(f"UNet output shape: {output.shape}")
    
    # Test AnomalyUNet
    anomaly_model = AnomalyUNet(n_channels=3)
    anomaly_model = anomaly_model.to(device)
    
    with torch.no_grad():
        recon, anomaly = anomaly_model(x)
    print(f"AnomalyUNet input shape: {x.shape}")
    print(f"AnomalyUNet reconstruction shape: {recon.shape}")
    print(f"AnomalyUNet anomaly map shape: {anomaly.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"UNet total parameters: {total_params:,}")
    
    total_params_anomaly = sum(p.numel() for p in anomaly_model.parameters())
    print(f"AnomalyUNet total parameters: {total_params_anomaly:,}")
