
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms, models

# ==========================================
# 1. 基本設定
# ==========================================
ROOT = Path(r"C:\MAX\data")
OUTPUT = Path(r"C:\MAX\outputs")
IMAGE_SIZE = 224
SEED = 42
MEMORY_SIZE = 1024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device, flush=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==========================================
# 2. 讀取圖片
# ==========================================
def load_image(path):
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def get_images(folder):
    return sorted(Path(folder).glob("*.png"))


# ==========================================
# 3. WideResNet50_2 特徵擷取器
# ==========================================
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = models.wide_resnet50_2(weights=weights)

        self.stem = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        f3 = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        features = torch.cat([f2, f3], dim=1)

        return features.permute(
            0, 2, 3, 1
        ).reshape(-1, features.shape[1])


# ==========================================
# 4. 建立正常圖片 Memory Bank
# ==========================================
@torch.inference_mode()
def build_memory(model, train_images):
    memory = []

    rng = np.random.default_rng(SEED)

    for i, path in enumerate(train_images):
        image = load_image(path).to(device)
        features = model(image).cpu()

        # 每張圖片隨機取部分 Patch
        count = min(10, len(features))

        selected = rng.choice(
            len(features),
            size=count,
            replace=False
        )

        memory.append(features[selected])

        if (i + 1) % 20 == 0:
            print(
                f"Building memory: {i+1}/{len(train_images)}",
                flush=True
            )

    memory = torch.cat(memory, dim=0)

    # 控制記憶庫大小
    if len(memory) > MEMORY_SIZE:
        selected = rng.choice(
            len(memory),
            size=MEMORY_SIZE,
            replace=False
        )
        memory = memory[selected]

    return memory.to(device)


# ==========================================
# 5. 計算圖片異常分數
# ==========================================
@torch.inference_mode()
def anomaly_score(model, path, memory):
    image = load_image(path).to(device)
    features = model(image)

    distances = []

    for chunk in features.split(128):
        dist = torch.cdist(
            chunk.float(),
            memory.float()
        )

        distances.append(
            dist.min(dim=1).values
        )

    nearest = torch.cat(distances)

    # 最大異常 Patch 分數
    return nearest.max().item()


# ==========================================
# 6. 執行單一類別
# ==========================================
def run_category(category):
    print(f"\n===== {category.upper()} =====", flush=True)

    category_path = ROOT / category

    train_images = get_images(
        category_path / "train" / "good"
    )

    if not train_images:
        raise RuntimeError("找不到正常訓練圖片")

    # 正常資料 80% Train、20% Validation
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(train_images))

    n_val = max(1, int(np.ceil(len(indices) * 0.2)))

    val_images = [
        train_images[i] for i in indices[:n_val]
    ]

    train_images = [
        train_images[i] for i in indices[n_val:]
    ]

    print("Train:", len(train_images))
    print("Validation:", len(val_images))

    # 官方 Test Set 保持不變
    test_images = []
    test_labels = []

    test_path = category_path / "test"

    for folder in sorted(test_path.iterdir()):
        if not folder.is_dir():
            continue

        for path in get_images(folder):
            test_images.append(path)
            test_labels.append(
                0 if folder.name == "good" else 1
            )

    if len(set(test_labels)) != 2:
        raise RuntimeError("Test 缺少正常或瑕疵圖片")

    print("Test:", len(test_images), flush=True)

    # 載入 ImageNet 預訓練特徵模型
    model = FeatureExtractor().to(device)
    model.eval()

    print("Building Memory Bank...", flush=True)
    memory = build_memory(model, train_images)

    # Validation 只包含正常圖片
    print("Evaluating validation...", flush=True)
    val_scores = [
        anomaly_score(model, p, memory)
        for p in val_images
    ]

    # 驗證集最大正常分數，作為保守閾值
    threshold = float(
        np.nextafter(max(val_scores), np.inf)
    )

    # Test 僅在模型與閾值確定後評估
    print("Evaluating test...", flush=True)
    test_scores = []

    for i, path in enumerate(test_images):
        score = anomaly_score(model, path, memory)
        test_scores.append(score)

        if (i + 1) % 10 == 0:
            print(
                f"Test: {i+1}/{len(test_images)}",
                flush=True
            )

    auc = roc_auc_score(
        test_labels,
        test_scores
    )

    print(f"\n{category} AUROC: {auc:.4f}", flush=True)

    # 儲存結果
    output_dir = OUTPUT / category
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "category": category,
        "model": "WideResNet50_2",
        "method": "PatchCore-style random memory baseline",
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "train_count": len(train_images),
        "validation_count": len(val_images),
        "test_count": len(test_images),
        "memory_size": len(memory),
        "validation_threshold": threshold,
        "test_auroc": float(auc),
        "test_samples": [
            {
                "path": str(p),
                "label": int(y),
                "anomaly_score": float(s)
            }
            for p, y, s in zip(
                test_images, test_labels, test_scores
            )
        ]
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)

    with open(
        output_dir / "split_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "train": [str(p) for p in train_images],
            "validation": [str(p) for p in val_images],
            "test": [str(p) for p in test_images]
        }, f, indent=2)

    print("Results saved:", output_dir, flush=True)


# ==========================================
# 7. 主程式
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        choices=["carpet", "bottle", "hazelnut"],
        default="carpet"
    )

    args = parser.parse_args()

    run_category(args.category)
