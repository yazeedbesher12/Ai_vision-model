"""
============================================================
WISBIN - MobileNetV3 Fine-Tuning
============================================================

Trains a 4-class waste classifier on crops captured by
test_wisbin_ai_full.py (keys 1/2/3/4).

CRITICAL DESIGN DECISION
------------------------
Normalization is baked INTO the model graph.

The exported ONNX takes RAW pixels in range 0..255 and does
the ImageNet mean/std scaling itself, as the first layers of
the network.

This means the OpenVINO conversion needs NO --mean_values and
NO --scale_values flags, which removes the single most common
cause of "great offline, garbage live" blobs.

Expected dataset layout:

    wisbin_dataset/
    ├── metal/
    ├── other/
    ├── paper/
    └── plastic/

============================================================
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# Their previous run crashed on a truncated JPEG.
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_SIZE = 224


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET INTEGRITY CHECK
# ============================================================

def scan_dataset(root):
    """Actually decode every image, not just verify() headers."""

    bad = []
    total = 0

    for folder, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            path = os.path.join(folder, name)
            total += 1

            try:
                with Image.open(path) as image:
                    image.convert("RGB").load()
            except Exception as error:
                bad.append((path, str(error)))

    return total, bad


# ============================================================
# MODEL WITH BAKED-IN NORMALIZATION
# ============================================================

class WisbinNet(nn.Module):
    """
    Wraps MobileNetV3.

    forward() expects a float tensor in range 0..255, NCHW, RGB.
    The scaling happens inside, so the exported graph is
    self-contained.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        weights = (
            MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        )

        self.backbone = mobilenet_v3_large(weights=weights)

        in_features = self.backbone.classifier[3].in_features

        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        # Stored as buffers so they travel with state_dict and export
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0,
        )

        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0,
        )

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return self.backbone(x)


# ============================================================
# LOAD DONOR WEIGHTS (e.g. EcoVision)
# ============================================================

def load_init_weights(model, path):
    """
    Loads a plain torchvision MobileNetV3 state_dict into
    WisbinNet, which nests it under .backbone.

    The final classifier layer is dropped whenever its class
    count differs, so a 10-class donor can seed a 4-class model.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    target = model.backbone.state_dict()

    cleaned = {}
    for key, value in checkpoint.items():
        name = key
        for prefix in ("module.", "backbone.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value

    loaded = 0
    skipped_shape = []
    skipped_missing = []

    final = {}
    for name, value in cleaned.items():
        if name not in target:
            skipped_missing.append(name)
            continue
        if tuple(target[name].shape) != tuple(value.shape):
            skipped_shape.append(
                f"{name}: donor {tuple(value.shape)} vs model "
                f"{tuple(target[name].shape)}"
            )
            continue
        final[name] = value
        loaded += 1

    model.backbone.load_state_dict(final, strict=False)

    print(f"Donor weights   : {path}")
    print(f"  tensors loaded  : {loaded} / {len(target)}")

    if skipped_shape:
        print(f"  shape mismatch  : {len(skipped_shape)} (expected for the head)")
        for line in skipped_shape:
            print(f"      {line}")

    if skipped_missing:
        print(f"  not in model    : {len(skipped_missing)}")
        for line in skipped_missing[:5]:
            print(f"      {line}")

    coverage = loaded / max(len(target), 1)

    if coverage < 0.80:
        print()
        print("WARNING: less than 80% of the model was initialised from")
        print("the donor file. The architecture probably does not match.")
        print("Stop and check before trusting the results.")
    else:
        print(f"  coverage        : {coverage * 100:.1f}%  OK")

    return loaded


# ============================================================
# TRANSFORMS
#
# ToTensor() gives 0..1, so we multiply by 255 to feed the
# model the same raw range it will see on the OAK-D.
# ============================================================

class ScaleTo255:
    def __call__(self, tensor):
        return tensor * 255.0


def build_transforms(train, bg_fill=127, jitter=0.12):
    """
    bg_fill matters: the crops have a uniform grey background.
    Rotating with the default black fill would invent black
    corners that never occur at inference time.
    """

    fill = (bg_fill, bg_fill, bg_fill)

    if train:
        return transforms.Compose([
            transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # Objects land at arbitrary angles under a top-down camera
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=180, fill=fill)], p=0.7
            ),
            # Kept mild: strong jitter would tint the grey background
            transforms.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter * 0.8,
                hue=jitter * 0.15,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
                fill=fill,
            ),
            transforms.ToTensor(),
            ScaleTo255(),
        ])

    return transforms.Compose([
        transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
        transforms.ToTensor(),
        ScaleTo255(),
    ])


# ============================================================
# SPLIT A FLAT FOLDER INTO TRAIN / VAL
# ============================================================

def stratified_split(dataset, val_ratio, seed):

    targets = np.array([label for _, label in dataset.samples])

    train_idx = []
    val_idx = []

    rng = np.random.RandomState(seed)

    for class_id in np.unique(targets):
        idx = np.where(targets == class_id)[0]
        rng.shuffle(idx)

        cut = max(1, int(round(len(idx) * val_ratio)))

        val_idx.extend(idx[:cut].tolist())
        train_idx.extend(idx[cut:].tolist())

    return sorted(train_idx), sorted(val_idx)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes):

    model.eval()

    loss_sum = 0.0
    correct = 0
    seen = 0

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss_sum += loss.item() * labels.size(0)
        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        seen += labels.size(0)

        for true_id, pred_id in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            confusion[true_id][pred_id] += 1

    return loss_sum / max(seen, 1), correct / max(seen, 1), confusion


def print_confusion(confusion, classes):

    width = max(len(name) for name in classes) + 2

    print()
    print("Confusion matrix (rows = true, cols = predicted)")
    print(" " * width + "".join(f"{name[:8]:>10s}" for name in classes))

    for i, name in enumerate(classes):
        row = "".join(f"{value:>10d}" for value in confusion[i])
        print(f"{name:<{width}s}{row}")

    print()
    print("Per-class recall:")

    for i, name in enumerate(classes):
        total = confusion[i].sum()
        recall = confusion[i][i] / total if total else 0.0
        print(f"  {name:<12s} {recall * 100:6.2f}%   ({confusion[i][i]}/{total})")

    print()


# ============================================================
# ONE TRAINING PHASE
# ============================================================

def run_phase(
    name,
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    scheduler,
    epochs,
    classes,
    best_state,
    checkpoint_path,
):

    best_accuracy = best_state["accuracy"]

    for epoch in range(1, epochs + 1):

        model.train()

        loss_sum = 0.0
        correct = 0
        seen = 0

        started = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = loss_sum / max(seen, 1)
        train_accuracy = correct / max(seen, 1)

        val_loss, val_accuracy, _ = evaluate(
            model, val_loader, device, criterion, len(classes)
        )

        marker = ""

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state["accuracy"] = val_accuracy

            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            marker = "  <-- best, saved"

        print(
            f"[{name}] epoch {epoch:2d}/{epochs}  "
            f"train_loss {train_loss:.4f}  train_acc {train_accuracy * 100:5.2f}%  "
            f"val_loss {val_loss:.4f}  val_acc {val_accuracy * 100:5.2f}%  "
            f"({time.time() - started:.0f}s){marker}"
        )

    return best_accuracy


# ============================================================
# MYRIAD-SAFE ACTIVATIONS
#
# The Myriad X compiler rejects the ONNX "HardSigmoid"
# operator that MobileNetV3 exports by default.
#
# These modules are mathematically identical but build the
# same curve out of ReLU6, which the compiler does support:
#
#   hardsigmoid(x) = relu6(x + 3) / 6
#   hardswish(x)   = x * hardsigmoid(x)
# ============================================================

class HardSigmoidSafe(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=False)

    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0


class HardSwishSafe(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=False)

    def forward(self, x):
        return x * (self.relu6(x + 3.0) / 6.0)


def make_myriad_safe(module):
    """Recursively swap Hardswish/Hardsigmoid for ReLU6 equivalents."""

    replaced = {"Hardswish": 0, "Hardsigmoid": 0}

    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, HardSwishSafe())
            replaced["Hardswish"] += 1
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, HardSigmoidSafe())
            replaced["Hardsigmoid"] += 1
        else:
            deeper = make_myriad_safe(child)
            for key in replaced:
                replaced[key] += deeper[key]

    return replaced


# ============================================================
# ONNX EXPORT
# ============================================================

def export_onnx(model, path, opset, myriad_safe=True):

    model.eval()
    model.cpu()

    if myriad_safe:
        probe = torch.randint(0, 256, (1, 3, MODEL_SIZE, MODEL_SIZE)).float()

        with torch.no_grad():
            before = model(probe).clone()

        counts = make_myriad_safe(model)
        model.eval()

        with torch.no_grad():
            after = model(probe)

        drift = (before - after).abs().max().item()

        print(
            f"Myriad-safe activations: "
            f"{counts['Hardswish']} Hardswish, "
            f"{counts['Hardsigmoid']} Hardsigmoid replaced"
        )
        print(f"  max output drift : {drift:.6f}  (should be ~0)")

        if drift > 1e-3:
            print("  WARNING: outputs changed. Do not trust this export.")

    dummy = torch.randint(0, 256, (1, 3, MODEL_SIZE, MODEL_SIZE)).float()

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"ONNX exported: {path}")
    print(f"  input  'input'  : [1, 3, {MODEL_SIZE}, {MODEL_SIZE}]  raw 0..255 RGB planar")
    print("  output 'logits' : [1, num_classes]  raw logits, apply softmax yourself")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3 on WISBIN crops"
    )

    parser.add_argument(
        "--data", default="wisbin_dataset",
        help="Folder with one subfolder per class",
    )
    parser.add_argument(
        "--test", default=None,
        help="Optional separate folder captured on a DIFFERENT day",
    )
    parser.add_argument("--out", default="wisbin_out")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=14)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze", type=int, default=4,
        help="How many trailing backbone blocks to unfreeze in phase 2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument(
        "--balanced-sampler", action="store_true",
        help="Oversample rare classes so every epoch sees all "
             "classes equally often (on top of class weights)",
    )
    parser.add_argument(
        "--weight-power", type=float, default=1.0,
        help="0 = ignore imbalance, 1 = full inverse-frequency "
             "weighting, 0.5 = half strength",
    )
    parser.add_argument(
        "--keep-hardswish", action="store_true",
        help="Keep native Hardswish/Hardsigmoid (Myriad X will reject them)",
    )
    parser.add_argument(
        "--init-weights", default=None,
        help="Donor .pth to start from, e.g. ecovision_mobilenetv3_ST.pth",
    )
    parser.add_argument(
        "--bg-fill", type=int, default=127,
        help="Background grey level used to fill rotated corners",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.12,
        help="ColorJitter strength; keep low for uniform backgrounds",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Skip ImageNet weights (offline testing only, hurts accuracy)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only scan the dataset for corrupt images, then exit",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    # --------------------------------------------------------
    # INTEGRITY
    # --------------------------------------------------------

    print("=" * 60)
    print("SCANNING DATASET")
    print("=" * 60)

    if not os.path.isdir(args.data):
        print(f"ERROR: folder not found: {args.data}")
        sys.exit(1)

    total, bad = scan_dataset(args.data)

    print(f"Images checked : {total}")
    print(f"Bad images     : {len(bad)}")

    for path, error in bad:
        print(f"  CORRUPT: {path}  ({error})")

    if bad:
        quarantine = os.path.join(args.out, "corrupt")
        os.makedirs(quarantine, exist_ok=True)

        for path, _ in bad:
            try:
                shutil.move(path, os.path.join(quarantine, os.path.basename(path)))
            except Exception:
                pass

        print(f"Moved {len(bad)} corrupt file(s) to {quarantine}")

    if args.check_only:
        return

    if total == 0:
        print("ERROR: no images found.")
        sys.exit(1)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_tf = build_transforms(True, args.bg_fill, args.jitter)
    eval_tf = build_transforms(False, args.bg_fill, args.jitter)

    train_base = datasets.ImageFolder(args.data, train_tf)
    val_base = datasets.ImageFolder(args.data, eval_tf)

    classes = train_base.classes
    num_classes = len(classes)

    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in train_base.samples:
        counts[label] += 1

    print()
    print("=" * 60)
    print("CLASSES")
    print("=" * 60)

    for i, name in enumerate(classes):
        print(f"  {i}: {name:<12s} {counts[i]:5d} images")

    if counts.min() < 30:
        print()
        print("WARNING: at least one class has fewer than 30 images.")
        print("Results will be unreliable. Aim for 150-250 per class.")

    imbalance = counts.max() / max(counts.min(), 1)
    if imbalance > 3:
        print()
        print(f"WARNING: class imbalance ratio is {imbalance:.1f}x.")
        print("Class weights are applied, but collecting more of the")
        print("rare classes works far better.")

    train_idx, val_idx = stratified_split(train_base, args.val_ratio, args.seed)

    train_set = Subset(train_base, train_idx)
    val_set = Subset(val_base, val_idx)

    print()
    print(f"Train images : {len(train_set)}")
    print(f"Val   images : {len(val_set)}")

    if args.test:
        test_base = datasets.ImageFolder(args.test, eval_tf)

        if test_base.classes != classes:
            print()
            print("ERROR: test folder class names do not match training set.")
            print(f"  train: {classes}")
            print(f"  test : {test_base.classes}")
            sys.exit(1)

        print(f"Test  images : {len(test_base)}  (separate session)")
    else:
        test_base = None
        print("Test  images : none (--test not given)")
        print()
        print("NOTE: without a separate-session test folder, the val")
        print("accuracy below is optimistic. Images from one session")
        print("are highly correlated.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0,
    )

    if args.balanced_sampler:

        # Probability of drawing each training image, set so that
        # every class is equally likely regardless of its size.
        train_targets = np.array(
            [train_base.samples[i][1] for i in train_idx]
        )

        train_counts = np.bincount(train_targets, minlength=num_classes)

        per_class = 1.0 / np.maximum(train_counts, 1)
        sample_weights = per_class[train_targets]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_targets),
            replacement=True,
        )

        train_loader = DataLoader(
            train_set, sampler=sampler, drop_last=False, **loader_args
        )

        print()
        print("Balanced sampler : ON")
        print(f"  images per epoch : {len(train_targets)}")
        print( "  each class now appears roughly equally often")

    else:
        train_loader = DataLoader(
            train_set, shuffle=True, drop_last=False, **loader_args
        )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    test_loader = (
        DataLoader(test_base, shuffle=False, **loader_args) if test_base else None
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL")
    print("=" * 60)
    print(f"Device : {device}")

    if use_cuda:
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU    : not available, training on CPU (slow)")

    print(f"Torch  : {torch.__version__}")

    use_imagenet = not args.no_pretrained and not args.init_weights

    model = WisbinNet(num_classes, pretrained=use_imagenet)

    if args.init_weights:
        if not os.path.isfile(args.init_weights):
            print(f"ERROR: donor file not found: {args.init_weights}")
            sys.exit(1)
        load_init_weights(model, args.init_weights)
    elif use_imagenet:
        print("Donor weights   : ImageNet (torchvision default)")
    else:
        print("Donor weights   : none (random init)")

    model = model.to(device)

    raw_weights = counts.sum() / (num_classes * counts)
    raw_weights = raw_weights ** args.weight_power

    weights = torch.tensor(raw_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    print(f"Class weights: {[round(float(w), 3) for w in weights]}"
          f"   (power {args.weight_power})")

    checkpoint_path = os.path.join(args.out, "wisbin_best.pth")
    best_state = {"accuracy": 0.0}

    # --------------------------------------------------------
    # PHASE 1 - HEAD ONLY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 1 - CLASSIFIER HEAD ONLY")
    print("=" * 60)

    for param in model.backbone.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.head_lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.head_epochs, 1)
    )

    run_phase(
        "head", model, train_loader, val_loader, device, criterion,
        optimizer, scheduler, args.head_epochs, classes,
        best_state, checkpoint_path,
    )

    # --------------------------------------------------------
    # PHASE 2 - UNFREEZE TAIL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"PHASE 2 - FINE-TUNE LAST {args.unfreeze} BLOCKS")
    print("=" * 60)

    blocks = list(model.backbone.features)
    unfreeze_from = max(0, len(blocks) - args.unfreeze)

    for index, block in enumerate(blocks):
        if index >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.finetune_lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.finetune_epochs, 1)
    )

    run_phase(
        "tune", model, train_loader, val_loader, device, criterion,
        optimizer, scheduler, args.finetune_epochs, classes,
        best_state, checkpoint_path,
    )

    # --------------------------------------------------------
    # FINAL EVALUATION ON BEST WEIGHTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    val_loss, val_accuracy, val_confusion = evaluate(
        model, val_loader, device, criterion, num_classes
    )

    print(f"Best val accuracy : {val_accuracy * 100:.2f}%   (loss {val_loss:.4f})")
    print_confusion(val_confusion, classes)

    test_accuracy = None

    if test_loader is not None:
        test_loss, test_accuracy, test_confusion = evaluate(
            model, test_loader, device, criterion, num_classes
        )

        print(f"Separate-session test accuracy : {test_accuracy * 100:.2f}%"
              f"   (loss {test_loss:.4f})")
        print_confusion(test_confusion, classes)

        print("This number is the honest one. Trust it over val.")

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------

    with open(os.path.join(args.out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    onnx_path = os.path.join(args.out, "wisbin_mobilenetv3.onnx")
    export_onnx(model, onnx_path, args.opset, myriad_safe=not args.keep_hardswish)

    summary = {
        "balanced_sampler": args.balanced_sampler,
        "weight_power": args.weight_power,
        "init_weights": args.init_weights,
        "bg_fill": args.bg_fill,
        "classes": classes,
        "counts": {name: int(counts[i]) for i, name in enumerate(classes)},
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
        "input": {
            "name": "input",
            "shape": [1, 3, MODEL_SIZE, MODEL_SIZE],
            "range": "0-255 raw uint8 values, RGB, planar (NCHW)",
            "normalization": "baked into the graph, do NOT pass mean/scale to MO",
        },
        "output": {"name": "logits", "note": "raw logits, apply softmax on host"},
    }

    with open(os.path.join(args.out, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  {checkpoint_path}")
    print(f"  {onnx_path}")
    print(f"  {os.path.join(args.out, 'classes.json')}")
    print(f"  {os.path.join(args.out, 'model_info.json')}")
    print()
    print("Next: convert the ONNX to a .blob WITHOUT mean/scale flags,")
    print("then verify the blob input reports dataType U8F before")
    print("running it on the OAK-D.")


if __name__ == "__main__":
    main()
