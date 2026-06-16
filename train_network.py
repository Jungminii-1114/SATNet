import timm
import os
import glob
import cv2
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from sample_network import SAAM, CFDM

import yaml

try: 
    import wandb
except ImportError:
    wandb = None

#root = "/Users/ijeongmin/🔥이정민🔥/가천대학교/CVIP/Mirror Detect/Dataset/MSD"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
config = load_config()
root = config["data_root"]

class MirrorDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, size: int = 256):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        name = os.path.basename(img_path).replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation = cv2.INTER_NEAREST)
        # mask는 일반 이미지가 아니라 class label map이기 때문에 INTER_NEAREST 사용해야함.
        # bilinear interpolate로 resize하면 중간값이 생긴다. 

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image = (image - mean) / std

        #mask = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)
        mask = torch.from_numpy((mask > 127).astype("int64"))

        return image, mask
    
class SwinTBackbone_T(nn.Module):
    def __init__(self, img_size=256, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size = img_size,
        )

    def forward(self, x):
        features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        return features
    

class SwinTBackbone_S(nn.Module):
    def __init__(self, img_size=512, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size = img_size,
        )

    def forward(self, x):
        features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        return features

class TrainableSATNetWithSwin(nn.Module):
    def __init__(self, img_size=256, pretrained=False):
        super().__init__()

        self.encoder = SwinTBackbone_S(img_size=img_size, pretrained=pretrained)

        self.saam2 = SAAM(embed_dim=384, num_heads=12)
        self.saam3 = SAAM(embed_dim=768, num_heads=24)

        self.cfdm3 = CFDM(channels=768, stage=3)
        self.cfdm2 = CFDM(channels=384, stage=2, prev_channels=768)
        self.cfdm1 = CFDM(channels=192, stage=1, prev_channels=384)
        self.cfdm0 = CFDM(channels=96, stage=0, prev_channels=192)

        self.seg_head0 = nn.Conv2d(96, 2, kernel_size=1)
        self.seg_head1 = nn.Conv2d(192, 2, kernel_size=1)
        self.seg_head2 = nn.Conv2d(384, 2, kernel_size=1)
        self.seg_head3 = nn.Conv2d(768, 2, kernel_size=1)

    def forward(self, x):
        x_flip = torch.flip(x, dims=[-1])

        f0, f1, f2, f3 = self.encoder(x)
        ff0, ff1, ff2, ff3 = self.encoder(x_flip)

        ff0 = torch.flip(ff0, dims=[-1])
        ff1 = torch.flip(ff1, dims=[-1])

        f2, ff2 = self.saam2(f2, ff2)
        f3, ff3 = self.saam3(f3, ff3)

        out3 = self.cfdm3(f3, ff3)
        out2 = self.cfdm2(f2, ff2, prev_output=out3)
        out1 = self.cfdm1(f1, ff1, prev_output=out2)
        out0 = self.cfdm0(f0, ff0, prev_output=out1)

        p3 = self.seg_head3(out3)
        p2 = self.seg_head2(out2)
        p1 = self.seg_head1(out1)
        p0 = self.seg_head0(out0)

        return [p0, p1, p2, p3]

# train_dataset = MirrorDataset(img_dir = f"{root}/train/image", mask_dir=f"{root}/train/mask", size=256)
# test_dataset = MirrorDataset(img_dir = f"{root}/test/image", mask_dir=f"{root}/test/mask", size=256)

train_dataset = MirrorDataset(img_dir = f"{root}/train/image", mask_dir=f"{root}/train/mask", size=config["img_size"])
test_dataset = MirrorDataset(img_dir = f"{root}/test/image", mask_dir=f"{root}/test/mask", size=config["img_size"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
#model = TrainableSATNetWithSwin(img_size=256, pretrained=False)
model = TrainableSATNetWithSwin(img_size=config["img_size"], pretrained=config["pretrained"]).to(device)

optimizer=torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
loss_weights = [1.25, 1.25, 1.0, 1.5] # P0 ~ P4
criterion = nn.CrossEntropyLoss()



def satnet_loss(preds, mask, weights):
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for pred, weight in zip(preds, weights):
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False,)
        total_loss = total_loss + weight * criterion(pred, mask)

    return total_loss

def poly_lr_lambda(current_iter):
    return max(0.0, 1.0 - current_iter / total_iters) ** 1.0


num_epochs=config["num_epochs"]
total_iters = num_epochs * len(train_loader)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

global_step = 0

for epoch in range(num_epochs):
    model.train()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader),)

    for image, mask in progress_bar:
        image = image.to(device)
        mask = mask.to(device)

        preds = model(image)
        loss = satnet_loss(preds, mask, config["loss_weights"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1

        progress_bar.set_postfix({
            "loss" : f"{loss.item():.4f}",
            "lr" : f"{scheduler.get_last_lr()[0]:.6f}",
            "step" : global_step,
        })

        if global_step % config["log_interval"] == 0:
            print(f"epoch={epoch+1}, step={global_step},"
                  f"loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}",
                  flush=True)

    print(f"Epoch : {epoch + 1}, loss: {loss.item():.4f}")