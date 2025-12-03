"""Small smoke training script that runs a single training step on synthetic data.

Usage (from repository root):
    python -m tests.smoke_train --model fasterrcnn_resnet50_fpn --num-classes 5
    # Or via CLI entry point:
    smoke-train --model fasterrcnn_resnet50_fpn --num-classes 5

This uses `pretrained=False` to avoid downloads and runs on CPU by default.
"""

import argparse
import logging

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from DeepLearning import train_models

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticDetectionDataset(Dataset):
    def __init__(self, num_samples=8, image_size=224, num_boxes=1):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_boxes = num_boxes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(3, self.image_size, self.image_size)
        # one box centered
        x1 = self.image_size * 0.25
        y1 = self.image_size * 0.25
        x2 = self.image_size * 0.75
        y2 = self.image_size * 0.75
        target = {
            "boxes": torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cpu")

    model = train_models.get_model(args.model, args.num_classes, pretrained=False)
    model.to(device)

    optimizer = SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    dataset = SyntheticDetectionDataset(num_samples=4, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model.train()
    imgs, tgts = next(iter(loader))
    imgs = [i.to(device) for i in imgs]
    tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

    optimizer.zero_grad()
    loss_dict = model(imgs, tgts)
    losses = sum(v for v in loss_dict.values())
    losses.backward()
    optimizer.step()

    logger.info(f"Smoke train complete. Loss: {losses.item():.4f}")


if __name__ == "__main__":
    main()
