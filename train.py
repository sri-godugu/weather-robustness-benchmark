import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import set_seed, accuracy_top1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results/checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # CIFAR-10 transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ResNet18 adapted for CIFAR-10 (change first conv + remove maxpool)
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_ld, desc=f"train epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        accs = []
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                accs.append(accuracy_top1(logits, y))
        test_acc = float(sum(accs) / len(accs))
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "seed": args.seed,
            }
            torch.save(ckpt, Path(args.outdir) / f"cifar10_resnet18_seed{args.seed}_best.pt")

        print(f"epoch={epoch} test_acc={test_acc:.4f} best={best_acc:.4f}")

    print("Done.")

if __name__ == "__main__":
    main()
