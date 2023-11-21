import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

BAND_COLORS = [
    'silver', 'white', 'blue', 'grey', 'violet', 'green', 
    'yellow', 'orange', 'red', 'gold', 'black', 'brown'
]
MAX_BANDS = 4

@dataclass
class CONFIG:
    log_level = logging.DEBUG
    data_path = Path('data')
    model_path = Path('what-the-ohm-4B.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = 42
    train_split = 0.80
    
    num_workers = 8
    batch_size = 8
    img_size = 256
    epochs = 15
    loss_weight = 5
    initial_lr = 0.0001
    train_log_interval = 5

# dataset to load resistor images and one hot encode bands
class ResistorDataset(Dataset):
    def __init__(self, data: pd.DataFrame, root_dir: Path, transform: v2.Transform = None):
        self.image_ids = data['image'].values
        self.root_dir = root_dir
        self.transform = transform
        self.encoded_labels = self.encode_labels(data['bands'])

    def encode_labels(self, labels: np.ndarray) -> torch.Tensor:
        encoded = pd.DataFrame()

        for i in range(1, MAX_BANDS + 1):
            for color in BAND_COLORS:
                encoded[f'band_{i}_{color}'] = labels.apply(lambda x: 1 if len(x) >= i and x[i - 1] == color else 0)
        return torch.tensor(encoded.values).type(torch.LongTensor)

    def load_img(self, idx: int) -> Image.Image:
        return Image.open(self.root_dir / self.image_ids[idx])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.load_img(idx)
        y = self.encoded_labels[idx]

        if self.transform:
            X = self.transform(X)

        return (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

class BCEWithLogitsLossWeighted(nn.Module):
    def __init__(self, weight, *args, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        self.weight = weight
    
    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        binary_labels = labels.bool()
        loss[binary_labels] *= labels[binary_labels] * self.weight
        return torch.mean(loss)

# init environment
def setup():
    # seed everything for reproducibility
    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)
    torch.manual_seed(CONFIG.random_seed)
    torch.cuda.manual_seed_all(CONFIG.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_name = datetime.now().strftime('logs/train_%Y%m%d%H%M%S.log')
    logging.basicConfig(
        level=CONFIG.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_name, mode='w'),
        ],
    )

    # log env/config information for sanity
    logging.debug(f'torch version: {torch.__version__}')
    logging.debug(f'{torch.cuda.device_count()} GPU(s) available')
    logging.debug(f'config: {CONFIG}')

# load training data from csv
def load_data(csv_path: str) -> pd.DataFrame:
    logging.debug(f'Reading data from {csv_path}')
    df = pd.read_csv(csv_path)
    df['bands'] = df['bands'].apply(lambda x: x.split(' '))

    # filter to only 4-band images
    df = df[df['image'].str.startswith('4-band')]

    logging.debug(f'Train shape: {df.shape}')
    logging.debug(f'Train columns: {df.columns}')
    return df

# get distinct classes using one-hot encoding
def get_classes(df: pd.DataFrame) -> List[str]:
    encoded = pd.DataFrame(index=df.index)

    for i in range(1, MAX_BANDS + 1):
        for color in BAND_COLORS:
            encoded[f'band_{i}_{color}'] = df['bands'].apply(lambda x: 1 if len(x) >= i and x[i - 1] == color else 0)

    classes = list(encoded.columns)
    logging.debug(f'{len(classes)} classes -> {classes}')
    return classes

# setup training and validation dataloaders
def get_dataloaders(df: pd.DataFrame, root_dir: Path) -> Tuple[DataLoader, DataLoader]:
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    all_ds = ResistorDataset(df, root_dir)
    total_samples = len(all_ds)
    train_size = int(total_samples * CONFIG.train_split)
    valid_size = total_samples - train_size
    train_ds, valid_ds = torch.utils.data.random_split(all_ds, [train_size, valid_size])

    logging.info(f'Train dataset: {len(train_ds)} samples')
    train_transform = v2.Compose([
        v2.Resize((CONFIG.img_size, CONFIG.img_size)),
        v2.CenterCrop(CONFIG.img_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=(-135.0, 135.0)),
        v2.ColorJitter(contrast=0.5, brightness=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])
    logging.debug(f'Using train transform: {train_transform}')
    train_ds.dataset.transform = train_transform
    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, num_workers=CONFIG.num_workers)

    logging.info(f'Valid dataset: {len(valid_ds)} samples')
    valid_ds.dataset.transform = v2.Compose([
        v2.Resize(CONFIG.img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])
    valid_loader = DataLoader(valid_ds, batch_size=CONFIG.batch_size, shuffle=False, num_workers=CONFIG.num_workers)

    return (train_loader, valid_loader)

# build neural network
def build_model(classes: List[str]) -> nn.Module:
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=len(classes))
    logging.debug(f'Using EfficientNet_v2_m')

    return model.to(CONFIG.device)

# model training step
def train_step(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, 
               scheduler: optim.lr_scheduler.LRScheduler = None) -> List[float]:
    losses = []
    model.train()

    for batch, (inputs, labels) in enumerate(dataloader):
        # forward pass
        inputs, labels = inputs.to(CONFIG.device), labels.to(CONFIG.device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        # backward pass + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % CONFIG.train_log_interval == 0:
            current_batch = (batch + 1) * len(inputs)
            logging.info(f'Training: Loss = {loss.item():>7f} [{current_batch:>5d}/{len(dataloader.dataset):>5d}]')
    
    if scheduler:
        scheduler.step()

    return losses

# calculate model accuracy
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (logits > threshold).float()
    correct = (preds == labels).sum().item()
    return correct / (labels.size(0) * labels.size(1))

# model validation step
def valid_step(dataloader: DataLoader, model: nn.Module, criterion: nn.Module):
    total_loss, total_accuracy = 0.0, 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(CONFIG.device), labels.to(CONFIG.device)
            logits = model(inputs)

            loss = criterion(logits, labels)
            total_loss += loss
            total_accuracy += calculate_accuracy(logits, labels)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    logging.info(f'Validation: Average Accuracy={(100 * avg_accuracy):>0.1f}%, Average Loss={avg_loss:>8f}')

def main():
    setup()

    # load data
    all_data = load_data(CONFIG.data_path / 'train.csv')
    train_loader, valid_loader = get_dataloaders(all_data, CONFIG.data_path)

    # build model
    classes = get_classes(all_data)
    model = build_model(classes)
    criterion = BCEWithLogitsLossWeighted(weight=CONFIG.loss_weight)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train model
    logging.debug("Started training model")
    start_time = time.time()
    losses = []
    for epoch in range(CONFIG.epochs):
        logging.info(f'Started epoch [{epoch+1}/{CONFIG.epochs}]')
        epoch_losses = train_step(train_loader, model, criterion, optimizer, scheduler)
        valid_step(valid_loader, model, criterion)
        losses.extend(epoch_losses)

    logging.debug(f"Model trained after {((time.time() - start_time) / 60):.2f} minute(s)")

    # export model
    logging.info(f"Saved model to {CONFIG.model_path}")
    torch.save(model.state_dict(), CONFIG.model_path)

if __name__ == '__main__':
    main()
