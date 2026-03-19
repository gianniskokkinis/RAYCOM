import os
from glob import glob
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class PairedImageDataset(Dataset):
    """
    Reads pairs of images from:
      - root/A/*_A.png  (RF heatmaps)
      - root/B/*_B.png  (optical renders)
    Matches the ids from the filenames (e.g., '3_A.png' with '3_B.png').
    """

    def __init__(self, root, image_size=256):
        super().__init__()
        self.path_A = os.path.join(root, "A")
        self.path_B = os.path.join(root, "B")

        a_files = sorted(glob(os.path.join(self.path_A, "*_A.png")))
        self.pairs = []
        for a in a_files:
            basename = os.path.basename(a)
            pair_id = basename.split("_A.")[0]
            b = os.path.join(self.path_B, f"{pair_id}_B.png")
            if os.path.exists(b):
                self.pairs.append((a, b))

        if not self.pairs:
            raise RuntimeError(f"No A/B pairs found under {root}")

        # Conversions: [0,1] -> [-1,1] for pix2pix-style training
        # this need for 256x256 pixels
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path = self.pairs[idx]
        img_A = Image.open(a_path).convert("RGB")
        img_B = Image.open(b_path).convert("RGB")
        return self.transform(img_A), self.transform(img_B)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    Simple UNet (similar to pix2pix) 3x256x256 -> 3x256x256
    """

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        # Down
        self.down1 = UNetBlock(in_channels, features, down=True)  # 256 -> 128
        self.down2 = UNetBlock(features, features * 2, down=True)  # 128 -> 64
        self.down3 = UNetBlock(features * 2, features * 4, down=True)  # 64 -> 32
        self.down4 = UNetBlock(features * 4, features * 8, down=True)  # 32 -> 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
        )  # 8 -> 4

        # Up
        self.up1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True)
        self.up2 = UNetBlock(features * 16, features * 4, down=False, use_dropout=True)
        self.up3 = UNetBlock(features * 8, features * 2, down=False)
        self.up4 = UNetBlock(features * 4, features, down=False)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.bottleneck(d4)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        out = self.up5(torch.cat([u4, d1], dim=1))
        return out


class PatchDiscriminator(nn.Module):
    """
    Simple PatchGAN discriminator for pix2pix
    Input: concatenation (A,B) -> 6 channels
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        # x: input (A), y: target ή generated (B)
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


def train(
    dataroot="dataset_ex1",
    checkpoint_dir="checkpoints",
    epochs=100,
    batch_size=4,
    lr=2e-4,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    dataset = PairedImageDataset(dataroot, image_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        for i, (real_A, real_B) in enumerate(loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            bs = real_A.size(0)

            # --------------------
            #  Train Discriminator
            # --------------------
            opt_D.zero_grad()
            with torch.no_grad():
                fake_B = G(real_A)

            pred_real = D(real_A, real_B)
            pred_fake = D(real_A, fake_B)

            valid = torch.ones_like(pred_real, device=device)
            fake = torch.zeros_like(pred_fake, device=device)

            loss_D_real = criterion_gan(pred_real, valid)
            loss_D_fake = criterion_gan(pred_fake, fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ----------------
            #  Train Generator
            # ----------------
            opt_G.zero_grad()
            fake_B = G(real_A)
            pred_fake = D(real_A, fake_B)

            loss_G_GAN = criterion_gan(pred_fake, valid)
            loss_G_L1 = criterion_l1(fake_B, real_B) * 100.0
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            if (i + 1) % 10 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] "
                    f"[Batch {i+1}/{len(loader)}] "
                    f"Loss_D: {loss_D.item():.4f} "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"(GAN: {loss_G_GAN.item():.4f}, L1: {loss_G_L1.item():.4f})"
                )

        # Model storage per epoch
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            G.state_dict(),
            os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"),
        )
        print(f"Saved checkpoint for epoch {epoch} to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pix2pix on Raycom dataset")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="dataset_ex1",
        help="Root folder with subfolders A and B (default: dataset_ex1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints",
        help="Folder to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    args = parser.parse_args()

    train(
        dataroot=args.dataroot,
        checkpoint_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )