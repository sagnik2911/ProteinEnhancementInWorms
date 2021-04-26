import torch.nn as nn


class CNNProteinEnhancer(nn.Module):
    def __init__(self):
        super(CNNProteinEnhancer, self).__init__()

        # First half: Downsampling from 512*512(1 channel) to a 4 * 4 vector (1024 channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # [512,512,1] -> [512,512,4]
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [512,512,4] -> [256,256,4]

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),  # [256,256,4] -> [256,
            # 256,16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [256,256,16] -> [128,128,16]

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            # [128,128,16] -> [128,128,64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [128,128,64] -> [64,64,64]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # [64,64,64] -> [64,64,128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [64,64,128] -> [32,32,128]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # [32,32,128] -> [32,32,256]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [32,32,256] -> [16,16,256]

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # [16,16,256] -> [16,16,512]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [16,16,256] -> [8,8,512]

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            # [8,8,512] -> [8,8,1024]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),  # [8,8,1024] -> [4,4,1024]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # [4,4,1024] -> [8,8,512]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [8,8,512] -> [16,16,256]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [16,16,256] -> [32,32,128]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [32,32,128] -> [64,64,64]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),  # [64,64,64] -> [128,128,32]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1),  # [128,128,32] -> [256,256,16]
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # [256,256,16] -> [512,512,4]
            nn.Sigmoid()
        )

    def forward(self, x):
        downscale = self.encoder(x)
        upscale = self.decoder(downscale)
        return upscale
