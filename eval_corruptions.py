import argparse
from pathlib import Path
import pandas as pd
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import accuracy_top1, mean_confidence_and_entropy
from corruptions.registry import CORRUPTIONS

def build_model(device):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_csv", type=str, default="results/robustness.csv")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    rows = []
    for cname in sorted(CORRUPTIONS.keys()):
        corrupt_fn = CORRUPTIONS[cname]
        for sev in [1, 2, 3, 4, 5]:
            acc_list, conf_list, ent_list = [], [], []
            for x, y in tqdm(test_ld, desc=f"{cname} sev{sev}"):
                x, y = x.to(device), y.to(device)
                x_cor = corrupt_fn(x, sev)
                logits = model(x_cor)
                acc_list.append(accuracy_top1(logits, y))
                conf, ent = mean_confidence_and_entropy(logits)
                conf_list.append(conf)
                ent_list.append(ent)

            row = {
                "corruption": cname,
                "severity": sev,
                "accuracy": float(sum(acc_list) / len(acc_list)),
                "mean_confidence": float(sum(conf_list) / len(conf_list)),
                "mean_entropy": float(sum(ent_list) / len(ent_list)),
            }
            rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
